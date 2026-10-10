# PyInstaller entry point; the package's __main__ uses relative imports, so it can't be the script itself.
from foresight.__main__ import main

if __name__ == "__main__":
    main()
