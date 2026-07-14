import threading
import webbrowser

from app import app


def open_browser() -> None:
    webbrowser.open("http://127.0.0.1:5001", new=0, autoraise=True)


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
