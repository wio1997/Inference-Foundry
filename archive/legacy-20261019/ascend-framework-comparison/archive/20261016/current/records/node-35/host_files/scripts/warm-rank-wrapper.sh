#!/usr/bin/env bash
set -u
ROLE=${1:?}; NODE=${2:?}
ROOT=/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015
case "$ROLE:$NODE" in p:33|p:34|p:35|p:37|d:36|d:41) ;; *) exit 2 ;; esac
LOG=$ROOT/logs/$ROLE-$NODE.engine.log
EXIT=$ROOT/logs/$ROLE-$NODE.exit
[ ! -e "$LOG" ] && [ ! -e "$EXIT" ] || exit 3
echo $$ > "$ROOT/logs/$ROLE-$NODE.wrapper.pid"
bash "$ROOT/scripts/warm-launch-$ROLE.sh" "$NODE" > "$LOG" 2>&1
RC=$?
echo "$RC" > "$EXIT"
exit "$RC"
