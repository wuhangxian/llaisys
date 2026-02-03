add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- ============================================================================
-- 【全局强制配置】(Global Configuration)
-- 下面的配置会应用到本项目的所有 Target，包括 includes 进来的文件
-- ============================================================================

set_languages("cxx17")
set_warnings("all", "error")

-- 【核心修复】强制开启 fPIC
-- 使用 {force = true} 强迫 xmake 在编译静态库时也加上这个参数
-- 只有加上这个，静态库才能被链接进动态库
if not is_plat("windows") then
    add_cxflags("-fPIC", {force = true})
    add_cxflags("-Wno-unknown-pragmas", {force = true})
end

-- ============================================================================
-- 【子模块引入】
-- ============================================================================

-- CPU (它会自动继承上面的全局 -fPIC 配置)
includes("xmake/cpu.lua")

-- NVIDIA
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- ============================================================================
-- 【Target 定义】
-- ============================================================================

target("llaisys-utils")
    set_kind("static")
    on_install(function(target) end) -- 禁止安装中间产物
    add_files("src/utils/*.cpp")
target_end()

target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    on_install(function(target) end) -- 禁止安装中间产物
    add_files("src/device/*.cpp")
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    on_install(function(target) end) -- 禁止安装中间产物
    add_files("src/core/*/*.cpp")
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")
    on_install(function(target) end) -- 禁止安装中间产物
    add_files("src/tensor/*.cpp")
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    on_install(function(target) end) -- 禁止安装中间产物
    add_files("src/ops/*/*.cpp")
target_end()

target("llaisys")
    set_kind("shared") -- 最终的动态库
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    add_files("src/llaisys/*.cc")
    
    -- 指定安装目录到 build/install，防止权限报错
    set_installdir("build/install")
    
    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            -- 拷贝生成的 .so 文件到 python 目录
            os.cp(target:targetfile(), "python/llaisys/libllaisys/")
        end
    end)
target_end()