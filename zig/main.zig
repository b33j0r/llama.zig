const std = @import("std");
const llama = @import("llama");

pub fn main() !u8 {
    llama.backendInit();
    defer llama.backendFree();

    return 0;
}