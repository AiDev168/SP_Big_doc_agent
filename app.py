"""Entry point:  streamlit run app.py"""
import localrag  # noqa: F401  (installs the offline guard before anything else is imported)
from localrag.ui.main import main

if __name__ == "__main__":
    main()
