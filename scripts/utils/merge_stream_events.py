import argparse
import pickle

from utu.ui.common import Event


def _yield_current_event(current_event):
    """Helper function to yield current event and return None."""
    if current_event is not None:
        yield current_event
    return None


def merged_event_stream(events: list[Event]):
    current_event = None

    for event in events:
        real_event = event["event"]

        # Handle raw events
        if real_event.type == "raw":
            data = real_event.data

            # Handle text and reason events (mergeable types)
            if data.type in ("text", "reason"):
                if isinstance(data.delta, list):
                    data.delta = "".join([item.text for item in data.delta])
                if current_event is None:
                    current_event = event
                elif current_event["event"].data.type == data.type:
                    current_event["event"].data.delta += data.delta
                    if len(current_event["event"].data.delta) > 600:
                        yield current_event
                        current_event = event
                else:
                    current_event["event"].data.delta += data.delta
                continue

            # Handle tool call events
            if data.type in ("tool_call", "tool_call_output"):
                yield from _yield_current_event(current_event)
                current_event = None
                yield event
                continue

            raise ValueError(f"Unsupported raw event data type: {data.type}")

        # Handle other event types that just need to flush current event
        if real_event.type in ("orchestra", "finish", "example"):
            yield from _yield_current_event(current_event)
            current_event = None
            yield event
            continue

        # Handle new event type with special processing
        if real_event.type == "new":
            yield from _yield_current_event(current_event)
            current_event = None
            if real_event.data.name.endswith(" (Agent)"):
                real_event.data.name = real_event.data.name[: -len(" (Agent)")]
            yield event
            continue

        raise ValueError(f"Unsupported event type: {real_event.type}")

    # Yield any remaining event
    if current_event is not None:
        yield current_event


def process_events(events: list[Event]):
    return list(merged_event_stream(events))


def save_events(events: list[Event], output: str):
    with open(output, "wb") as f:
        pickle.dump(events, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    args = parser.parse_args()
    with open(args.events, "rb") as f:
        events = pickle.load(f)
    events = process_events(events)
    output_name = args.output or args.events.replace(".pkl", "_merged.pkl")
    save_events(events, output_name)
