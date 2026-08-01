#!/bin/sh
# Regenerate compile_commands.json for a header-only tree.
#
# CMake only emits entries for real translation units, and this library
# has none, so the database is built directly from the header list.
# Paths are absolute and resolved at run time: re-run this after moving
# or cloning the repository.

set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/compile_commands.json"
compiler=${CXX:-c++}
std=${LINALG_STD:-c++17}

rm -f "$out"   # may be a symlink into the build directory after a cmake configure

{
    printf '[\n'
    first=1
    find "$root/include" "$root/src" "$root/tests" \
        \( -name '*.hpp' -o -name '*.cpp' \) 2>/dev/null | sort |
    while IFS= read -r header; do
        [ "$first" -eq 1 ] || printf ',\n'
        first=0
        printf '  {\n'
        printf '    "directory": "%s",\n' "$root"
        printf '    "file": "%s",\n' "$header"
        printf '    "output": "%s",\n' "${header%.hpp}.o"
        printf '    "arguments": [\n'
        printf '      "%s",\n' "$compiler"
        printf '      "-std=%s",\n' "$std"
        case "$header" in
            *.hpp) printf '      "-x", "c++-header",\n' ;;
            *)     printf '      "-x", "c++",\n' ;;
        esac
        printf '      "-I", "%s/include",\n' "$root"
        printf '      "-Wall",\n'
        printf '      "-Wextra",\n'
        printf '      "-Wpedantic",\n'
        printf '      "-c", "%s"\n' "$header"
        printf '    ]\n'
        printf '  }'
    done
    printf '\n]\n'
} > "$out"

printf 'wrote %s (%d entries)\n' "$out" \
    "$(find "$root/include" "$root/src" "$root/tests" \
        \( -name '*.hpp' -o -name '*.cpp' \) 2>/dev/null | wc -l)"
