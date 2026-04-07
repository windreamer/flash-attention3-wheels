"""Verify that .pyd files inside a Windows wheel export the required PyInit symbols.

Usage:
    python verify_wheel_exports.py <wheel_path>

Exits with code 1 if any .pyd is missing its expected PyInit_<module> export.
"""

import os
import struct
import sys
import tempfile
import zipfile


def get_pe_exports(path: str) -> list[str]:
    """Extract exported function names from a PE (DLL/PYD) file."""
    with open(path, "rb") as f:
        # MZ header
        if f.read(2) != b"MZ":
            return []
        f.seek(0x3C)
        pe_offset = struct.unpack("<I", f.read(4))[0]

        # PE signature
        f.seek(pe_offset)
        if f.read(4) != b"PE\0\0":
            return []

        # COFF header
        _machine = struct.unpack("<H", f.read(2))[0]
        num_sections = struct.unpack("<H", f.read(2))[0]
        f.read(12)  # timestamp, symbol table pointer, symbol count
        optional_hdr_size = struct.unpack("<H", f.read(2))[0]
        f.read(2)  # characteristics

        # Optional header – determine PE32 vs PE32+
        optional_start = f.tell()
        magic = struct.unpack("<H", f.read(2))[0]
        if magic == 0x10B:  # PE32
            export_dir_offset = 96
        elif magic == 0x20B:  # PE32+ (64-bit)
            export_dir_offset = 112
        else:
            return []

        # Export directory RVA & size
        f.seek(optional_start + export_dir_offset)
        export_rva = struct.unpack("<I", f.read(4))[0]
        _export_size = struct.unpack("<I", f.read(4))[0]
        if export_rva == 0:
            return []

        # Section headers
        f.seek(optional_start + optional_hdr_size)
        sections: list[tuple[int, int, int, int]] = []
        for _ in range(num_sections):
            f.read(8)  # name
            virtual_size = struct.unpack("<I", f.read(4))[0]
            virtual_address = struct.unpack("<I", f.read(4))[0]
            raw_size = struct.unpack("<I", f.read(4))[0]
            raw_offset = struct.unpack("<I", f.read(4))[0]
            f.read(16)  # relocs, linenums, counts, characteristics
            sections.append((virtual_address, virtual_size, raw_offset, raw_size))

        def rva_to_file_offset(rva: int) -> int | None:
            for va, vs, ro, _rs in sections:
                if va <= rva < va + vs:
                    return ro + (rva - va)
            return None

        # Parse export directory table
        export_offset = rva_to_file_offset(export_rva)
        if export_offset is None:
            return []

        f.seek(export_offset + 24)  # NumberOfNames
        num_names = struct.unpack("<I", f.read(4))[0]
        f.seek(export_offset + 32)  # AddressOfNames RVA
        names_rva = struct.unpack("<I", f.read(4))[0]

        names_offset = rva_to_file_offset(names_rva)
        if names_offset is None:
            return []

        names: list[str] = []
        for i in range(num_names):
            f.seek(names_offset + i * 4)
            name_rva = struct.unpack("<I", f.read(4))[0]
            name_offset = rva_to_file_offset(name_rva)
            if name_offset is None:
                continue
            f.seek(name_offset)
            raw = b""
            while True:
                ch = f.read(1)
                if ch in (b"\0", b""):
                    break
                raw += ch
            names.append(raw.decode("ascii", errors="replace"))

        return names


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <wheel_path>", file=sys.stderr)
        return 2

    wheel_path = sys.argv[1]
    if not os.path.isfile(wheel_path):
        print(f"Error: file not found: {wheel_path}", file=sys.stderr)
        return 2

    errors: list[str] = []
    with zipfile.ZipFile(wheel_path) as zf:
        pyd_files = [n for n in zf.namelist() if n.endswith(".pyd")]
        if not pyd_files:
            print("No .pyd files found in wheel – nothing to verify.")
            return 0

        tmpdir = tempfile.mkdtemp()
        try:
            for pyd_name in pyd_files:
                module_name = os.path.splitext(os.path.basename(pyd_name))[0]
                expected_sym = f"PyInit_{module_name}"

                pyd_path = zf.extract(pyd_name, tmpdir)
                exports = get_pe_exports(pyd_path)

                if expected_sym in exports:
                    print(f"  OK: {expected_sym} found in {pyd_name}")
                else:
                    print(
                        f"  FAIL: {expected_sym} NOT found in {pyd_name}"
                        f" (exports: {exports})"
                    )
                    errors.append(pyd_name)
        finally:
            # Clean up extracted files
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    if errors:
        print(
            f"\nERROR: {len(errors)} .pyd file(s) missing required PyInit exports!"
        )
        return 1

    print("\nAll .pyd exports verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
