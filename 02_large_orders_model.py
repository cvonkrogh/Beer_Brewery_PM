import importlib.util
from pathlib import Path


_small_model_path = Path(__file__).with_name("02_small_orders_model.py")
_small_spec = importlib.util.spec_from_file_location("small_orders_02", _small_model_path)
_small_module = importlib.util.module_from_spec(_small_spec)
_small_spec.loader.exec_module(_small_module)

main = _small_module.main


if __name__ == "__main__":
    main()
