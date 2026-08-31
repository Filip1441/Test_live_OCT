#ifdef _WIN32
#include "file_dialog.hpp"
#include <windows.h>
#include <commdlg.h>
#include <vector>
#include <string>

namespace oct {

std::string FileDialog::open_file(const std::string& title, const std::vector<std::string>& extensions) {
    wchar_t filename[MAX_PATH] = {0};

    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = L"HDF5 Files (*.hdf5;*.h5)\0*.hdf5;*.h5\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrDefExt = L"hdf5";

    if (GetOpenFileNameW(&ofn)) {
        char mb_path[MAX_PATH * 4];
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, mb_path, sizeof(mb_path), NULL, NULL);
        return std::string(mb_path);
    }
    return "";
}

} // namespace oct
#endif
