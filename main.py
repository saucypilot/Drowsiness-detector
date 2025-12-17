from app import DrowsinessDetectorApp


def main() -> None:
    """Entry point for the drowsiness detector application."""
    app = DrowsinessDetectorApp()
    app.run()


if __name__ == "__main__":
    main()
