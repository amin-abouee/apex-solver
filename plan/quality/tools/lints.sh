#!/bin/bash
# usage: lints.sh <repo-root> <out.clippy.json> [cargo args, e.g. --workspace --lib --bins]
# Runs clippy with the structural/perf lint set used by the audit. Production code only
# when called with --lib --bins. Does not modify the repository.
root=$1; out=$2; shift 2
cd "$root" || exit 1
cargo clippy "$@" --message-format=json -- \
 -W clippy::too_many_lines -W clippy::cognitive_complexity -W clippy::large_enum_variant \
 -W clippy::needless_pass_by_value -W clippy::redundant_clone -W clippy::clone_on_ref_ptr \
 -W clippy::implicit_clone -W clippy::needless_collect -W clippy::trivially_copy_pass_by_ref \
 -W clippy::large_types_passed_by_value -W clippy::unnecessary_wraps -W clippy::cast_possible_truncation \
 -W clippy::cast_precision_loss -W clippy::cast_sign_loss -W clippy::float_cmp -W clippy::indexing_slicing \
 -W clippy::struct_excessive_bools -W clippy::fn_params_excessive_bools -W clippy::format_collect \
 -W clippy::or_fun_call -W clippy::significant_drop_tightening -W clippy::collection_is_never_read \
 -W clippy::branches_sharing_code -W clippy::manual_memcpy -W clippy::inefficient_to_string \
 -W clippy::large_stack_arrays -W clippy::missing_const_for_fn -W clippy::option_if_let_else \
 -W clippy::redundant_closure_for_method_calls -W clippy::similar_names -W clippy::many_single_char_names \
 -W clippy::match_same_arms -W clippy::needless_pass_by_ref_mut -W clippy::unused_self -W clippy::rc_buffer \
 -W clippy::box_collection -W clippy::vec_box -W clippy::mutex_atomic -W clippy::future_not_send \
 2>/dev/null > "$out"
