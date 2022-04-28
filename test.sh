set -eo pipefail

OUTCOME=$(notacommand | cat || (echo "\nthis failed" > /dev/tty && exit 1)
)
