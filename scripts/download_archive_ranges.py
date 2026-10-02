"""Download bounded HTTP ranges after observed large-transfer disconnections."""
import shutil
import subprocess
from pathlib import Path


def download_ranges(url, part, size, proxy_port, write_report, chunk_bytes):
    offset = part.stat().st_size if part.exists() else 0
    assert offset < size
    while offset < size:
        end = min(offset + chunk_bytes, size) - 1
        chunk = Path(str(part) + '.range_{}'.format(offset))
        assert not chunk.exists(), 'Preserve and inspect a previous failed range before resuming.'
        write_report('DOWNLOADING_RANGE', completed_bytes=offset, range_start=offset, range_end=end)
        result = subprocess.run([
            'curl', '--fail', '--location', '--show-error', '--silent', '--http1.1',
            '--proxy', 'socks5h://127.0.0.1:{}'.format(proxy_port), '--connect-timeout', '30',
            '--speed-time', '180', '--speed-limit', '1024', '--max-time', '240',
            '--range', '{}-{}'.format(offset, end), '--output', str(chunk),
            '--write-out', '%{http_code} %{size_download} %{speed_download}\n', url,
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        write_report('RANGE_RETURNED', curl_exit_code=result.returncode,
                     curl_summary=result.stdout.strip(), curl_stderr=result.stderr)
        if result.returncode != 0:
            write_report('FAILED_DOWNLOAD', failed_range=str(chunk))
            raise SystemExit(result.returncode)
        assert result.stdout.split()[0] == '206', result.stdout
        assert chunk.stat().st_size == end - offset + 1
        with chunk.open('rb') as source, part.open('ab') as destination:
            shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
        offset = part.stat().st_size
        assert offset == end + 1
        chunk.unlink()
        write_report('RANGE_COMPLETE', completed_bytes=offset)
