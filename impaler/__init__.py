from pkgutil import extend_path
import os
__path__ = extend_path(__path__, __name__)
__path__.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
