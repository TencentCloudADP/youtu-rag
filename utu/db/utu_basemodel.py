from pydantic import BaseModel


class UTUBaseModel(BaseModel):
    def update(self, **kwargs):
        """
        Update the evaluation sample with the given keyword arguments.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get(self, key, default=None):
        """
        Get the value of the specified key, or return default if not found.
        """
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an EvaluationSample from a dictionary.
        """
        return cls(**data)

    def as_dict(self) -> dict:
        # only contain fields that are not None
        return {k: v for k, v in self.model_dump().items() if v is not None}
