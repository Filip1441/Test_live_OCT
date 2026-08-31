import time
from fair_core import fair_load, fair_edit_metadata, fair_save

CONFIG_PATH = '/Users/filip/Library/Application Support/LocalViewer/config.json'

print("=== KROK 1: Wybór obiektu (tylko metadane / szybki podgląd) ===")
obj_preview = fair_load(
    CONFIG_PATH,
    ("Data_Type", "raw_projections"),
    metadata_only=True
)
if obj_preview is None:
    print("Anulowano wybór obiektu.")
    exit()

print("=== KROK 2: Wybór referencji (tylko metadane / szybki podgląd) ===")
ref_preview = fair_load(
    CONFIG_PATH,
    ("Data_Type", "raw_projections"),
    metadata_only=True
)
if ref_preview is None:
    print("Anulowano wybór referencji.")
    exit()

print("=== KROK 3: Edycja metadanych ===")
parent_uuids = [obj_preview.metadata["UUID"], ref_preview.metadata["UUID"]]
new_metadata = fair_edit_metadata(parent_uuids, config=CONFIG_PATH)

if new_metadata is None:
    print("Edycja metadanych została anulowana.")
    exit()

print("=== KROK 4: Automatyczne pobranie pełnych danych (bez GUI) ===")
print("Trwa wczytywanie pełnych macierzy 3D z pliku dyskowego...")
obj_full = fair_load(CONFIG_PATH, uuid=obj_preview.metadata["UUID"])
ref_full = fair_load(CONFIG_PATH, uuid=ref_preview.metadata["UUID"])

print("=== KROK 5: Dummy Obliczenia ===")
print("Rozpoczęto rekonstrukcję Phase-OCT (Volume3DEngine)...")
time.sleep(1.0)
print("Obliczanie widma i transformata DTFT...")
time.sleep(1.0)
print("Uciąglanie fazy Miguela 2D (Union-Find)...")
time.sleep(1.0)
print("Obliczenia zakończone sukcesem!")

dummy_result = {"ph_reconstructed": obj_full.data[list(obj_full.data.keys())[0]][:, :, 0] if obj_full.data else "dummy"}

print("=== KROK 6: Zapis wyniku ===")
result = fair_save(CONFIG_PATH, data=dummy_result, metadata=new_metadata)
print(f"Dane zostały zapisane do systemu FAIR pomyślnie. Nowy UUID: {result['uuid']}")
