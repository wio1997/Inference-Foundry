"""Isolated host library build. Never installs a library or launches a model."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tarfile
import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
assert not torch.npu.is_initialized()
INSTALLED = Path(torch_npu.__file__).parent
assert hashlib.sha256((INSTALLED / "lib/libtorch_npu.so").read_bytes()).hexdigest() == (
    "83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842")
SRC = ROOT / "source"
SRC.mkdir()
ENV = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", MAX_JOBS="4")


def status(phase, **extra):
    row = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               NPU_initialized=False, model_request=False, installed_library_modified=False, **extra)
    target = ROOT / "build_state.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(target)


def command(phase, argv, cwd):
    status(phase, argv=argv)
    with (ROOT / (phase + ".stdout.log")).open("xb") as stdout, (
        ROOT / (phase + ".stderr.log")).open("xb") as stderr:
        proc = subprocess.run(argv, cwd=cwd, env=ENV, stdout=stdout, stderr=stderr,
                              timeout=5400)
    if proc.returncode:
        raise RuntimeError(phase + " exit=" + str(proc.returncode))


try:
    assert shutil.disk_usage(ROOT).free >= 15 * 1024**3
    archives = json.loads((ROOT / "archive_identity.json").read_text())
    for row in archives:
        archive = ROOT / row["name"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["sha256"]
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                assert not member.name.startswith("/") and ".." not in Path(member.name).parts
            tar.extractall(SRC)
    project = SRC / "pytorch-5dd8ef3f9b375b5ae4a83538d5785754148c3302"
    mappings = {
        "op-plugin-8b9c8534fa41eff367a41c155843daa530ab3a08": "third_party/op-plugin",
        "fmt-123913715afeb8a437e6388b4473fcc4753e1c9a": "third_party/fmt",
        "json-87cda1d6646592ac5866dc703c8e1839046a6806": "third_party/nlohmann",
        "dvm-af14bb28675c447196b43b2c573a21d576007a26": "third_party/dvm/dvm",
        "Tensorpipe-d8a18ac35924bfc56b36a138868e3c71ff5f203f": "third_party/Tensorpipe",
    }
    for source, destination in mappings.items():
        target = project / destination
        if target.exists():
            assert target.is_dir() and not list(target.iterdir())
            target.rmdir()
        (SRC / source).rename(target)

    ops = project / "third_party/op-plugin/op_plugin/ops/opapi"
    shutil.copyfile(ROOT / "mc2_diagnostic_capability.h", ops / "mc2_diagnostic_capability.h")
    edits = []
    for name, operation in (("MoeDistributeDispatchV2KernelOpApi.cpp", "Dispatch"),
                            ("MoeDistributeCombineKernelV2OpApi.cpp", "Combine")):
        path = ops / name
        before = path.read_bytes()
        text = before.decode()
        old = 'check_aclnn_kernel_available("aclnnMoeDistribute' + operation + 'V4")'
        new = "glm_mc2_diag." + operation.lower() + "_available()"
        new = new.replace("glm_mc2_diag.", "glm_mc2_diag::")
        include = '#include "mc2_diagnostic_capability.h"\n'
        assert text.count(old) == 1
        after = text.replace(old, new).replace(
            '#include "op_plugin/utils/op_api_common.h"\n',
            '#include "op_plugin/utils/op_api_common.h"\n' + include)
        assert after.replace(new, old).replace(include, "") == text
        path.write_text(after)
        edits.append(dict(file=name, before_sha256=hashlib.sha256(before).hexdigest(),
            after_sha256=hashlib.sha256(after.encode()).hexdigest(),
            executable_body_identical_after_reversing_capability_selection=True))
    (ROOT / "native_candidate_identity.json").write_text(json.dumps(edits, indent=2) + "\n")

    # Original Tensorpipe shared library is reused, with its exact pinned headers.
    cmake_path = project / "CMakeLists.txt"
    cmake = cmake_path.read_text()
    start = cmake.index('if (DEFINED BUILD_TENSORPIPE)\n  add_definitions')
    end = cmake.index('\nif (DEFINED BUILD_LIBTORCH)', start)
    original_block = cmake[start:end]
    replacement = ('if (DEFINED BUILD_TENSORPIPE)\n'
        '  add_definitions(-DUSE_RPC_FRAMEWORK)\nendif()\n')
    cmake = cmake[:start] + replacement + cmake[end:]
    old_link = '${PROJECT_SOURCE_DIR}/build/packages/torch_npu/lib/libtensorpipe.so'
    cmake = cmake.replace(old_link, str(INSTALLED / "lib/libtensorpipe.so"))
    assert cmake.replace(replacement, original_block, 1).replace(
        str(INSTALLED / "lib/libtensorpipe.so"), old_link) == cmake_path.read_text()
    cmake_path.write_text(cmake)
    (ROOT / "build_support_identity.json").write_text(json.dumps(dict(
        original_cmake_block=original_block, replacement=replacement,
        reused_tensorpipe_sha256=hashlib.sha256((INSTALLED / "lib/libtensorpipe.so").read_bytes()).hexdigest(),
        tensorpipe_headers_commit="d8a18ac35924bfc56b36a138868e3c71ff5f203f",
        tensorpipe_implementation_unchanged=True), indent=2) + "\n")

    toolkit = Path("/usr/local/Ascend/ascend-toolkit/latest")
    acl_inc = project / "third_party/acl/inc"
    acl_inc.mkdir(parents=True, exist_ok=True)
    for name in ("acl", "graph", "ge"):
        target = acl_inc / name
        source = toolkit / "include" / name
        assert source.is_dir(), str(source)
        # Archive contains an incomplete acl/ops tree. Preserve existing source
        # files and add only missing toolkit headers inside this isolated clone.
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        elif not target.is_symlink():
            for item in source.rglob("*"):
                destination = target / item.relative_to(source)
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(item, target_is_directory=item.is_dir())
        assert (acl_inc / "acl/acl.h").is_file()
    libraries = project / "third_party/acl/libs"
    for name in ("libhccl.so", "libascendcl.so", "libacl_op_compiler.so", "libge_runner.so", "libgraph.so"):
        source = toolkit / "lib64" / name
        assert source.exists(), str(source)
        (libraries / name).symlink_to(source)

    command("codegen", ["bash", "generate_code.sh", os.sys.executable, "2.10.0"], project)
    build = project / "build"
    build.mkdir()
    output = ROOT / "output"
    output.mkdir()
    torch_dir = Path(torch.__file__).parent
    args = [shutil.which("cmake"), "-S", str(project), "-B", str(build), "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_OPPLUGIN=on", "-DBUILD_NEW_HEADER=on",
        "-DBUILD_TENSORPIPE=on", "-DGLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "-DPYTORCH_INSTALL_DIR=" + str(torch_dir), "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_path("include"),
        "-DTORCH_VERSION=2.10.0", "-DTORCHNPU_INSTALL_LIBDIR=" + str(output),
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(output),
        "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=" + str(output),
        "-DCMAKE_CXX_FLAGS=-I" + str(toolkit / "include") + " -I" + str(toolkit / "pkg_inc")]
    command("configure", args, project)
    command("compile", [shutil.which("cmake"), "--build", str(build), "--parallel", "4"], project)
    assert not torch.npu.is_initialized()
    library = output / "libtorch_npu.so"
    assert library.is_file()
    status("compiled", library=str(library), bytes=library.stat().st_size,
           sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
           model_correctness=False, matched_AB=False, E2E_gain=False)
except BaseException as error:
    status("failed", error=repr(error), model_correctness=False, matched_AB=False)
    raise
