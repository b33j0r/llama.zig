const std = @import("std");
const llama = @import("llama");

pub fn main() !u8 {
    const allocator = std.heap.c_allocator;

    var backend = llama.Llama.init();
    defer backend.deinit();

    const loader = llama.ModelLoader.default();
    var model = try loader.load();
    defer model.deinit();

    var ctx = try llama.Context.init(&model, .{
        .n_ctx = 1024 * 2,
        .n_batch = 1024 * 2,
    });
    defer ctx.deinit();

    var sampler = try llama.Sampler.default();
    defer sampler.deinit();

    const pipeline = llama.Pipeline{
        .allocator = allocator,
        .ctx = &ctx,
        .sampler = &sampler,
    };

    var it = try pipeline.generate(
        \\JSON list of 25 fictional characters for a space opera. JSON only.
    );
    defer it.deinit();

    while (it.nextIgnoreErrors()) |part| {
        std.debug.print("{s}", .{part});
    }

    std.debug.print("\n", .{});
    return 0;
}
