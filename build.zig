const std = @import("std");

const Opts = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    use_accelerate: bool,
    use_metal: bool, // ObjC shim only in this 0.15 flow; ship default.metallib yourself if you enable it.
};

const LlamaCppDep = struct {
    b: *std.Build,
    opts: Opts,

    pub const root_dir = ".";
    pub const ggml_dir = "ggml";
    pub const include_dir = "include";

    pub fn init(b: *std.Build, opts: Opts) LlamaCppDep {
        return .{ .b = b, .opts = opts };
    }

    pub fn compile(self: *LlamaCppDep) *std.Build.Step.Compile {
        const b = self.b;

        const lib = b.addLibrary(.{
            .name = "llama_zig",
            .root_module = b.createModule(.{
                // only used to carry macros/flags; code lives in C/C++
                .root_source_file = b.path("zig/root.zig"),
                .target = self.opts.target,
                .optimize = .ReleaseFast,
            }),
            .linkage = .static,
        });

        // Includes & common macros
        lib.addIncludePath(b.path(include_dir));
        lib.addIncludePath(b.path(ggml_dir ++ "/include"));
        lib.addIncludePath(b.path(ggml_dir ++ "/src/ggml-cpu"));
        lib.addIncludePath(b.path(ggml_dir ++ "/src"));
        lib.addIncludePath(b.path(root_dir ++ "/src"));

        if (self.opts.target.result.os.tag == .macos and self.opts.target.result.cpu.arch == .aarch64) {
            lib.root_module.addCMacro("GGML_USE_K_QUANTS", "0");
            lib.root_module.addCMacro("GGML_USE_LLAMAFILE", "0");
        } else {
            lib.root_module.addCMacro("GGML_USE_K_QUANTS", "1");
        }
        lib.root_module.addCMacro("GGML_VERSION", "\"llama-zig-build\"");
        lib.root_module.addCMacro("GGML_COMMIT", "\"local\"");
        lib.root_module.addCMacro("GGML_USE_CPU", "1");

        // ggml core - just add the files directly
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml.c"), .flags = &.{} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-alloc.c"), .flags = &.{} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-backend.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-backend-reg.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-opt.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-quants.c"), .flags = &.{} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-threading.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/gguf.cpp"), .flags = &.{"-std=c++17"} });

        // ggml-cpu backend files
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/amx/amx.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/amx/mmq.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/binary-ops.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/ggml-cpu.c"), .flags = &.{} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/ggml-cpu.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/hbm.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/ops.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/quants.c"), .flags = &.{} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/repack.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/traits.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/unary-ops.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/vec.cpp"), .flags = &.{"-std=c++17"} });

        // Architecture-specific ggml-cpu files
        const target_cpu = self.opts.target.result.cpu.arch;
        switch (target_cpu) {
            .aarch64 => {
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/arm/cpu-feats.cpp"), .flags = &.{"-std=c++17"} });
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/arm/quants.c"), .flags = &.{} });
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/arm/repack.cpp"), .flags = &.{"-std=c++17"} });
            },
            .x86_64 => {
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/x86/cpu-feats.cpp"), .flags = &.{"-std=c++17"} });
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/x86/quants.c"), .flags = &.{} });
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/x86/repack.cpp"), .flags = &.{"-std=c++17"} });
            },
            .powerpc64, .powerpc64le => {
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/powerpc/cpu-feats.cpp"), .flags = &.{"-std=c++17"} });
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/powerpc/quants.c"), .flags = &.{} });
            },
            .riscv64 => {
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/riscv/quants.c"), .flags = &.{} });
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/riscv/repack.cpp"), .flags = &.{"-std=c++17"} });
            },
            .wasm32, .wasm64 => {
                lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-cpu/arch/wasm/quants.c"), .flags = &.{} });
            },
            else => {},
        }

        // llama front-end files
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-impl.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-io.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-model.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-vocab.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-arch.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-adapter.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-batch.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-chat.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-context.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-cparams.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-grammar.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-graph.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-hparams.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-kv-cache.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-kv-cache-iswa.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-memory.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-memory-hybrid.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-memory-recurrent.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-mmap.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-model-loader.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-model-saver.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-quant.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/llama-sampling.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/unicode.cpp"), .flags = &.{"-std=c++17"} });
        lib.addCSourceFile(.{ .file = b.path(root_dir ++ "/src/unicode-data.cpp"), .flags = &.{"-std=c++17"} });

        // Accelerate (macOS)
        if (self.opts.use_accelerate and self.opts.target.result.os.tag == .macos) {
            lib.root_module.addCMacro("GGML_USE_ACCELERATE", "1");
            lib.linkFramework("Accelerate");
        }

        // Metal (ObjC shim only on 0.15)
        if (self.opts.use_metal and self.opts.target.result.os.tag == .macos) {
            lib.root_module.addCMacro("GGML_USE_METAL", "1");
            lib.addCSourceFile(.{ .file = b.path(ggml_dir ++ "/src/ggml-metal.m"), .flags = &.{"-fobjc-arc"} });
            lib.linkFramework("Metal");
            lib.linkFramework("Foundation");
        }

        // C++ runtime
        switch (self.opts.target.result.os.tag) {
            .windows => {},
            .macos => lib.linkSystemLibrary("c++"),
            else => lib.linkSystemLibrary("stdc++"),
        }

        b.installArtifact(lib);
        return lib;
    }
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_accelerate = b.option(bool, "accelerate", "Enable Apple Accelerate on macOS") orelse false;
    const use_metal = b.option(bool, "metal", "Enable Metal backend on macOS (shader supplied at runtime)") orelse false;

    const opts = Opts{
        .target = target,
        .optimize = optimize,
        .use_accelerate = use_accelerate,
        .use_metal = use_metal,
    };

    // 1) low-level C/C++ lib (llama.cpp + ggml)
    var llama_dep = LlamaCppDep.init(b, opts);
    const llama_lib = llama_dep.compile();

    // 2) Zig module wrapper
    const llama_zig_mod = b.addModule("llama", .{
        .root_source_file = b.path("zig/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    llama_zig_mod.addIncludePath(b.path(LlamaCppDep.include_dir));
    llama_zig_mod.addIncludePath(b.path(LlamaCppDep.ggml_dir ++ "/include"));
    llama_zig_mod.addIncludePath(b.path(LlamaCppDep.ggml_dir ++ "/src"));
    llama_zig_mod.addIncludePath(b.path(LlamaCppDep.root_dir ++ "/src"));
    llama_zig_mod.linkLibrary(llama_lib);

    // 3) demo exe — imports module; links ONLY llama_zig
    const demo = b.addExecutable(.{
        .name = "demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "llama", .module = llama_zig_mod }},
        }),
    });
    switch (target.result.os.tag) {
        .windows => {},
        .macos => demo.linkSystemLibrary("c++"),
        else => demo.linkSystemLibrary("stdc++"),
    }
    b.installArtifact(demo);

    const run = b.addRunArtifact(demo);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run demo").dependOn(&run.step);

    // 4) tests — link the single llama lib too
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    tests.addIncludePath(b.path(LlamaCppDep.include_dir));
    tests.addIncludePath(b.path(LlamaCppDep.ggml_dir ++ "/include"));
    tests.addIncludePath(b.path(LlamaCppDep.ggml_dir ++ "/src"));
    tests.addIncludePath(b.path(LlamaCppDep.root_dir ++ "/src"));
    tests.linkLibrary(llama_lib);
    switch (target.result.os.tag) {
        .windows => {},
        .macos => tests.linkSystemLibrary("c++"),
        else => tests.linkSystemLibrary("stdc++"),
    }
    b.step("test", "Run tests").dependOn(&b.addRunArtifact(tests).step);
}
