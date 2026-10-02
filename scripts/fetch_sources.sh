#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$PROJECT/third_party"
git clone --no-checkout https://github.com/supertyd/XTrack.git "$PROJECT/third_party/XTrack"
git -C "$PROJECT/third_party/XTrack" checkout --detach 8a606f00c5e98b5393254f6ee1891447b662407f
git clone --no-checkout https://github.com/chenxin-dlut/SUTrack.git "$PROJECT/third_party/SUTrack"
git -C "$PROJECT/third_party/SUTrack" checkout --detach d65052d1ba3fcf55010e1fb3665ee6616c139a2c
