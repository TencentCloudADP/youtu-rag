from utu.utils import FileUtils

WEB_URL = "http://www.pthxx.com/b_audio/pthxx_com_mp3/01_langdu/02.mp3"
LOCAL_URL = "/Users/frankshi/Downloads/02.mp3"


def test_ext():
    assert FileUtils.get_file_ext(WEB_URL) == ".mp3"
    assert FileUtils.is_web_url(WEB_URL)

    assert FileUtils.get_file_ext(LOCAL_URL) == ".mp3"
    assert not FileUtils.is_web_url(LOCAL_URL)


def test_md5():
    md5_web = FileUtils.get_file_md5(WEB_URL)
    md5_local = FileUtils.get_file_md5(LOCAL_URL)
    print(md5_web, md5_local)
    assert md5_web == md5_local
