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

    var handler = Handler{};
    const pipeline = llama.Pipeline{
        .self = &handler,
        .on_progress = &Handler.onProgress,
        .on_complete = &Handler.onComplete,
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

const Handler = struct {
    data: usize = 1337,
    pub fn onProgress(_: *anyopaque, text: []const u8) void {
        std.debug.print("{s}", .{text});
    }
    pub fn onComplete(ptr: *anyopaque, text: []const u8) void {
        const self: *Handler = @ptrCast(@alignCast(ptr));
        std.debug.print("\n{s}\n", .{text});
        std.debug.print("data: {d}\n", .{self.data});
    }
};
