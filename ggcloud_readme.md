# Kết nối dữ liệu RAV-21 trên Google Cloud

Tài liệu này dành cho thành viên đã được cấp quyền vào project Google Cloud của nhóm. Dữ liệu nằm trên persistent disk gắn vào VM GPU; không nằm trong Git và VM không có IP public.

## Thông tin hạ tầng

| Thành phần | Giá trị |
|---|---|
| Project ID | `project-0c6ab625-3efb-4fe3-934` |
| VM | `rav-21-nuscenes-t4-spot` |
| Zone | `asia-southeast1-b` |
| Kết nối | SSH qua Identity-Aware Proxy (IAP) |
| Data disk | `rav-21-nuscenes-data`, 800 GB |
| Điểm mount | `/data` |
| Source code trên VM | `/opt/rav21` |

VM dùng một GPU T4 Spot. VM có thể bị Google thu hồi khi thiếu capacity; watchdog thử khởi động lại mỗi 5 phút và pipeline tự resume từ checkpoint sau khi VM boot.

## 1. Chuẩn bị máy cá nhân

Cài đặt:

- Google Cloud CLI (`gcloud`).
- Visual Studio Code và extension **Remote - SSH** nếu muốn duyệt dữ liệu bằng VS Code.
- OpenSSH client. Windows 10/11 thường đã có sẵn tại `C:\Windows\System32\OpenSSH\ssh.exe`.

Đăng nhập bằng đúng Google account đã được cấp quyền:

```powershell
gcloud auth login
gcloud config set project project-0c6ab625-3efb-4fe3-934
gcloud auth list
```

Nếu máy có nhiều Google account, chọn account được cấp quyền:

```powershell
gcloud config set account YOUR_EMAIL
```

## 2. Kết nối lần đầu bằng terminal

Chạy lệnh sau. Lần đầu `gcloud` sẽ tạo SSH key và đăng ký public key lên project:

```powershell
gcloud compute ssh rav-21-nuscenes-t4-spot `
  --project=project-0c6ab625-3efb-4fe3-934 `
  --zone=asia-southeast1-b `
  --tunnel-through-iap
```

Trên macOS/Linux, thay dấu backtick bằng `\` hoặc viết lệnh trên một dòng.

Sau khi kết nối thành công, ghi lại Linux username của mình để dùng cho VS Code:

```bash
whoami
ls -la /data
```

Không dùng username `sonla` trong cấu hình của thành viên khác. Mỗi người dùng username do lần kết nối `gcloud compute ssh` đầu tiên tạo ra.

## 3. Mở `/data` bằng VS Code

### Windows

Tìm thư mục cài Google Cloud SDK:

```powershell
gcloud info --format="value(installation.sdk_root)"
```

Mở file `%USERPROFILE%\.ssh\config` và thêm cấu hình dưới đây. Thay `<LINUX_USER>`, `<WINDOWS_USER>` và, nếu cần, đường dẫn SDK theo kết quả lệnh phía trên.

```sshconfig
Host rav21-t4
    HostName rav-21-nuscenes-t4-spot
    User <LINUX_USER>
    IdentityFile C:/Users/<WINDOWS_USER>/.ssh/google_compute_engine
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 4
    ProxyCommand "C:/Users/<WINDOWS_USER>/AppData/Local/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe" "-S" "C:/Users/<WINDOWS_USER>/AppData/Local/Google/Cloud SDK/google-cloud-sdk/lib/gcloud.py" compute start-iap-tunnel rav-21-nuscenes-t4-spot %p --listen-on-stdin --project=project-0c6ab625-3efb-4fe3-934 --zone=asia-southeast1-b --verbosity=warning
```

Kiểm tra alias trước khi mở VS Code:

```powershell
ssh rav21-t4 "whoami; ls -la /data"
code --remote ssh-remote+rav21-t4 /data
```

Có thể chọn **Remote-SSH: Connect to Host...** → `rav21-t4` trong Command Palette thay cho lệnh `code`.

### macOS/Linux

Thêm vào `~/.ssh/config`; thay `<LINUX_USER>` và thay `/absolute/path/to/gcloud` bằng kết quả của `which gcloud`:

```sshconfig
Host rav21-t4
    HostName rav-21-nuscenes-t4-spot
    User <LINUX_USER>
    IdentityFile ~/.ssh/google_compute_engine
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 4
    ProxyCommand /absolute/path/to/gcloud compute start-iap-tunnel rav-21-nuscenes-t4-spot %p --listen-on-stdin --project=project-0c6ab625-3efb-4fe3-934 --zone=asia-southeast1-b --verbosity=warning
```

Sau đó chạy:

```bash
ssh rav21-t4 'whoami; ls -la /data'
code --remote ssh-remote+rav21-t4 /data
```

## 4. Các thư mục dữ liệu

| Đường dẫn | Nội dung | Cách sử dụng |
|---|---|---|
| `/data/nuscenes/dataset` | nuScenes trainval gốc | Chỉ đọc; chứa ảnh chưa làm mờ |
| `/data/nuscenes/models` | Model EgoBlur | Không sửa hoặc thay file |
| `/data/nuscenes/logs` | Log do pipeline ghi | Dùng để kiểm tra tiến độ |
| `/data/nuscenes/work` | File tạm riêng của pipeline | Không truy cập hoặc sao chép |
| `/data/rav21-data/derived/previews/trainval` | Video preview đã xử lý | Dùng để review kết quả |
| `/data/rav21-data/scenes/trainval/<scene>` | Manifest, quality, privacy và sync report | Đọc báo cáo theo scene |
| `/data/rav21-data/catalog` | Catalog đầu ra | Chỉ đọc |
| `/data/<folder-mới>` | Dữ liệu mới do thành viên xử lý | Được tạo trực tiếp dưới `/data` |
| `/opt/rav21` | Source code đang chạy trên VM | Không sửa trực tiếp khi production chạy |

Một số artifact do service tạo có thể chỉ cho owner đọc. Nếu gặp `Permission denied`, không dùng `sudo`, `chmod` hoặc copy vòng qua thư mục khác; gửi chính xác đường dẫn cần đọc cho người quản trị để cấp quyền phù hợp.

### Tạo thư mục output của thành viên

Mỗi thành viên được tạo thư mục mới trực tiếp dưới `/data`. Nên đặt tên có Linux username để tránh trùng tên. Không ghi kết quả vào dataset nguồn hoặc thư mục output của production:

```bash
mkdir -p "/data/${USER}-data"
mkdir -p "/data/${USER}-data/my-processing-output"
```

Kiểm tra quyền ghi và dung lượng còn trống trước khi chạy:

```bash
test -r /data/nuscenes/dataset && echo "dataset: readable"
test -w /data && echo "/data: writable"
df -h /data
```

Thư mục gốc `/data` dùng mode `1777` với sticky bit, tương tự `/tmp`: mọi tài khoản đã đăng nhập hợp lệ có thể tạo thư mục mới trực tiếp bên trong, nhưng không thể xóa hoặc đổi tên thư mục cấp đầu của người khác. Các thư mục mới thuộc về người tạo. Khi cần hai thành viên cùng ghi vào cùng một thư mục con, owner của thư mục cần cấp group/ACL phù hợp hoặc liên hệ người quản trị.

## 5. Kiểm tra trạng thái an toàn

Từ máy cá nhân:

```powershell
gcloud compute instances describe rav-21-nuscenes-t4-spot `
  --project=project-0c6ab625-3efb-4fe3-934 `
  --zone=asia-southeast1-b `
  --format="value(status)"
```

Sau khi SSH vào VM:

```bash
df -h /data
find /data/nuscenes/dataset -maxdepth 2 -type d | head
find /data/rav21-data/scenes/trainval -mindepth 1 -maxdepth 1 -type d | wc -l
tail -n 50 /data/nuscenes/logs/trainval-service.log
```

Ngoài thư mục mới do chính thành viên tạo dưới `/data`, chỉ dùng các lệnh đọc. Không stop/start service, xóa/di chuyển dữ liệu nguồn, chạy lại pipeline hoặc thay đổi output production nếu chưa phối hợp với người đang vận hành. Production dùng lock `/data/rav21-data/.trainval.lock` để ngăn hai tiến trình cùng ghi.

## 6. Xử lý lỗi thường gặp

### `403` hoặc thiếu `iap.tunnelInstances.accessViaIAP`

Kiểm tra `gcloud auth list` đang active đúng account được cấp quyền. Hai account của nhóm đã được cấp `roles/iap.tunnelResourceAccessor`; nếu dùng account khác cần nhờ quản trị viên cấp quyền tương ứng.

### `Connection timed out` hoặc VS Code vẫn trỏ tới IP cũ

VM không có IP public. Dùng alias `rav21-t4` với `ProxyCommand` IAP ở trên; không dùng entry SSH cũ chứa `HostName` là địa chỉ IP.

### VM ở trạng thái `TERMINATED`

Đây có thể là Spot preemption. Watchdog kiểm tra mỗi 5 phút và thử start lại trong cùng zone. Nếu zone chưa có T4, các lần thử có thể tiếp tục thất bại cho tới khi capacity quay lại.

### Có cảnh báo tăng tốc NumPy khi mở IAP tunnel

Cảnh báo về NumPy chỉ ảnh hưởng tốc độ truyền của IAP tunnel, không làm sai dữ liệu và không ngăn SSH hoạt động.

### `Permission denied` bên trong `/data`

Không tự đổi owner hoặc permission. Gửi đường dẫn bị lỗi cho quản trị viên; quyền truy cập sẽ được cấp ở mức tối thiểu cần thiết để tránh lộ dữ liệu ảnh gốc.

## 7. Quy tắc dữ liệu

- Dữ liệu nuScenes gốc có thể chứa khuôn mặt và biển số chưa làm mờ. Chỉ sử dụng trong phạm vi dự án và không chia sẻ ra ngoài.
- Không commit dữ liệu, model, log hoặc video vào Git.
- Chỉ ghi kết quả vào thư mục mới do thành viên tạo trực tiếp dưới `/data`, ví dụ `/data/${USER}-data`; không ghi vào `/data/nuscenes` hoặc `/data/rav21-data`.
- Không tải toàn bộ dataset về máy cá nhân nếu chưa được thống nhất.
- Ưu tiên làm việc với preview đã ẩn danh và các report JSON.
- Khi không chắc một thao tác có ghi dữ liệu hay ảnh hưởng production hay không, dừng lại và hỏi người vận hành.
