const std = @import("std");

pub const c = @cImport({
    @cDefine("LLAMA_STATIC", "1");
    @cInclude("llama.h");
    @cInclude("ggml.h");
});

/// One-time process init/teardown for llama.cpp backends.
pub fn backendInit() void {
    c.llama_backend_init();
    c.ggml_backend_load_all();

    const n_dev = c.ggml_backend_dev_count();
    std.debug.assert(n_dev > 0);
    var i: usize = 0;
    while (i < n_dev) : (i += 1) {
        const info = c.ggml_backend_reg_get(i);
        const name = c.ggml_backend_reg_name(info);
        std.log.info("llama.cpp ggml backend {s} (id={d}) available\n", .{ name, i });
    }
}

pub fn backendFree() void {
    c.llama_backend_free();
}

/// Small RAII-ish wrappers so user code stays tidy.
pub const Model = struct {
    ptr: *c.llama_model,

    pub fn load(alloc: std.mem.Allocator, path: []const u8) !Model {
        const pz = try alloc.dupeZ(u8, path);
        defer alloc.free(pz);

        const params = c.llama_model_default_params();

        const m = c.llama_load_model_from_file(@as([*c]const u8, @ptrCast(pz.ptr)), params);
        if (m == null) return error.LoadModelFailed;
        return .{ .ptr = m.? };
    }

    pub fn deinit(self: *Model) void {
        c.llama_free_model(self.ptr);
    }

    pub fn nVocab(self: *const Model) i32 {
        const vocab = c.llama_model_get_vocab(self.ptr.?);
        return c.llama_n_vocab(vocab);
    }

    pub fn tokenEos(self: *const Model) c.llama_token {
        const vocab = c.llama_model_get_vocab(self.ptr.?);
        return c.llama_token_eos(vocab);
    }
};

pub const Ctx = struct {
    ptr: ?*c.llama_context,
    model_ptr: ?*c.llama_model, // keep raw C pointer to query vocab/eos

    pub fn create(model: *const Model, n_ctx: u32) !Ctx {
        var cparams = c.llama_context_default_params();
        if (n_ctx > 0) cparams.n_ctx = n_ctx;

        const ctx = c.llama_new_context_with_model(model.ptr, cparams);
        if (ctx == null) return error.CreateContextFailed;

        return .{ .ptr = ctx, .model_ptr = model.ptr };
    }

    pub fn deinit(self: *Ctx) void {
        if (self.ptr) |p| c.llama_free(p);
        self.ptr = null;
        self.model_ptr = null;
    }

    pub const Token = c.llama_token;

    pub const TokenBuf = struct {
        items: []Token,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *TokenBuf) void {
            self.allocator.free(self.items);
            self.* = undefined;
        }
    };

    /// Tokenize UTF-8 `text` with the model's vocab.
    /// `add_special`: usually false for hand-built prompts.
    /// `parse_special`: false for raw user text; true only if you embed `<|...|>` control tokens.
    pub fn tokenize(
        self: *Ctx,
        allocator: std.mem.Allocator,
        text: []const u8,
        add_special: bool,
        parse_special: bool,
    ) !TokenBuf {
        std.debug.assert(self.model_ptr != null);
        const vocab = c.llama_model_get_vocab(self.model_ptr.?);
        if (vocab == null) return error.NoVocab;

        // Pass 1: probe — negative means "needed size".
        const null_tokens = @as([*c]Token, @ptrFromInt(0));
        const text_len = @as(i32, @intCast(text.len));

        var need = c.llama_tokenize(
            vocab,
            @as([*c]const u8, @ptrCast(text.ptr)),
            text_len,
            null_tokens,
            0,
            add_special,
            parse_special,
        );
        if (need == std.math.minInt(i32)) return error.TokenizeFailed;
        if (need < 0) need = -need;

        if (need == 0) {
            return TokenBuf{ .items = try allocator.alloc(Token, 0), .allocator = allocator };
        }

        // Pass 2: allocate + fill
        var out = try allocator.alloc(Token, @as(usize, @intCast(need)));
        const wrote = c.llama_tokenize(
            vocab,
            @as([*c]const u8, @ptrCast(text.ptr)),
            text_len,
            @as([*c]Token, @ptrCast(out.ptr)),
            @as(i32, @intCast(out.len)),
            add_special,
            parse_special,
        );

        if (wrote <= 0) {
            allocator.free(out);
            return error.TokenizeFailed;
        }
        if (wrote != @as(i32, @intCast(out.len))) {
            out = try allocator.realloc(out, @as(usize, @intCast(wrote)));
        }

        return TokenBuf{ .items = out, .allocator = allocator };
    }

    /// Greedy decode: evaluate prompt, then argmax tokens until EOS or `max_new_tokens`.
    pub fn generateGreedy(
        self: *Ctx,
        alloc: std.mem.Allocator,
        prompt: []const u8,
        max_new_tokens: i32,
        writer: anytype,
    ) !void {
        std.debug.assert(self.model_ptr != null);

        // 1) tokenize prompt (usually add BOS once)
        var tb = try self.tokenize(alloc, prompt, false, false);
        defer tb.deinit();

        // 2) evaluate full prompt in chunks
        const batch_size: usize = 64;
        var i: usize = 0;
        while (i < tb.items.len) : (i += @min(tb.items.len - i, batch_size)) {
            const n = @as(i32, @intCast(@min(tb.items.len - i, batch_size)));
            const batch = c.llama_batch_get_one(&tb.items[i], n);

            // IMPORTANT: do not write to batch.pos when it is NULL.
            // If pos == NULL and seq_id == NULL, llama_decode tracks positions automatically.

            if (c.llama_decode(self.ptr.?, batch) != 0)
                return error.EvalFailed;
        }

        // 3) stream tokens
        var produced: i32 = 0;
        const eos = getTokenEos(self.model_ptr.?);
        while (produced < max_new_tokens) : (produced += 1) {
            const next = try greedyPick(self);
            if (next == eos) break;

            try printTokenPiece(self.model_ptr.?, next, writer);

            // feed back one token; let positions auto-track (pos == NULL)
            var t = next;
            const batch = c.llama_batch_get_one(&t, 1);
            if (c.llama_decode(self.ptr.?, batch) != 0)
                return error.EvalFailed;
        }
    }
};

/// Argmax over the last logits.
fn greedyPick(ctx: *const Ctx) !c.llama_token {
    const logits_ptr = c.llama_get_logits(ctx.ptr.?);
    if (logits_ptr == null) return error.NoLogits;

    const vocab_n = @as(usize, @intCast(getNVocab(ctx.model_ptr.?)));
    var best_id: usize = 0;
    var best: f32 = -3.4028235e38;

    var i: usize = 0;
    while (i < vocab_n) : (i += 1) {
        const v = logits_ptr[i];
        if (v > best) {
            best = v;
            best_id = i;
        }
    }
    return @as(c.llama_token, @intCast(best_id));
}

fn getNVocab(model_ptr: *c.llama_model) i32 {
    const vocab = c.llama_model_get_vocab(model_ptr);
    return c.llama_n_vocab(vocab);
}

fn getTokenEos(model_ptr: *c.llama_model) c.llama_token {
    if (@hasDecl(c, "llama_token_eos")) {
        const vocab = c.llama_model_get_vocab(model_ptr);
        return c.llama_token_eos(vocab);
    }
    return 2;
}

/// Write a token’s byte-piece to an output stream.
fn printTokenPiece(model_ptr: *c.llama_model, tok: c.llama_token, writer: anytype) !void {
    var buf: [8 * 1024]u8 = undefined;
    const vocab = c.llama_model_get_vocab(model_ptr);
    const n = c.llama_token_to_piece(
        vocab,
        tok,
        &buf,
        @as(i32, @intCast(buf.len)),
        0,
        true,
    );
    if (n <= 0) return;
    try writer.writeAll(buf[0..@as(usize, @intCast(n))]);
}
