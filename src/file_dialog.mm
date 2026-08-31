#import <Cocoa/Cocoa.h>
#include "file_dialog.hpp"
#include <iostream>

namespace oct {

std::string FileDialog::open_file(const std::string& title, const std::vector<std::string>& extensions) {
    @autoreleasepool {
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setTitle:[NSString stringWithUTF8String:title.c_str()]];
        [panel setMessage:[NSString stringWithUTF8String:title.c_str()]];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];

        NSMutableArray* allowedTypes = [NSMutableArray array];
        for (const auto& ext : extensions) {
            [allowedTypes addObject:[NSString stringWithUTF8String:ext.c_str()]];
        }
        [panel setAllowedFileTypes:allowedTypes];

        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] firstObject];
            if (url) {
                return std::string([[url path] UTF8String]);
            }
        }
        return "";
    }
}

} // namespace oct
