const std = @import("std");
const llama = @import("llama");

pub fn main() !u8 {
    const allocator = std.heap.c_allocator;

    var backend = llama.Llama.init();
    defer backend.deinit();

    const loader = llama.ModelLoader.default();
    var model = try loader.load();
    defer model.deinit();

    var ctx = try llama.Context.init(&model, 1024 * 8);
    defer ctx.deinit();

    var sampler = try llama.Sampler.default();
    defer sampler.deinit();

    const pipeline = llama.Pipeline(
        onProgress,
        onComplete,
    ){
        .allocator = allocator,
        .ctx = &ctx,
        .sampler = &sampler,
        .max_tokens = 1024 * 2,
    };

    pipeline.generate("Make a list of fictional character names for a video game about space trading and piracy. Respond in JSON only") catch |err| {
        std.debug.print("Error during generation: {any}\n", .{err});
    };

    return 0;
}

fn onProgress(token: []const u8) void {
    std.debug.print("{s}", .{token});
}

fn onComplete(full: []const u8) void {
    std.debug.print("\n---\nFull output:\n{s}\n", .{full});
}
