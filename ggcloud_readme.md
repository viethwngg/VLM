# RAV-21 — Google Cloud & Data Access

Tài liệu dành cho thành viên RAV-21 đã được cấp quyền Google Cloud. Dữ liệu
nuScenes và artifacts nằm trên persistent disk `/data` gắn vào VM GPU.

Không lưu raw dataset vào Git.

## 1. Infrastructure Overview

| Component | Value |
| --- | --- |
| GCP Project | `project-0c6ab625-3efb-4fe3-934` |
| VM | `rav-21-nuscenes-t4-spot` |
| Zone | `asia-southeast1-b` |
| Access | SSH via IAP |
| GPU | NVIDIA T4 Spot |
| Data Disk | `rav-21-nuscenes-data` — 800 GB |
| Mount Point | `/data` |
| Project Source | `/opt/rav21` |

> **Lưu ý**
>
> - VM không có public IP.
> - Không SSH bằng IP; luôn dùng Identity-Aware Proxy (IAP).
> - Spot VM có thể bị Google preempt.
> - Watchdog thử start lại VM mỗi 5 phút. Sau khi VM boot, pipeline production
>   resume từ checkpoint.

## 2. Prerequisites

Cài đặt các công cụ sau trên máy cá nhân:

- [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install)
- OpenSSH client
- Visual Studio Code
- Extension **Remote - SSH** cho VS Code (không bắt buộc nếu chỉ dùng terminal)

Windows 10/11 thường có OpenSSH tại:

```text
C:\Windows\System32\OpenSSH\ssh.exe
```

### Cài Google Cloud CLI trên Windows

Nếu PowerShell báo `gcloud is not recognized`, tải và chạy
[Google Cloud CLI installer chính thức](https://cloud.google.com/sdk/docs/install-sdk#windows).
Giữ tùy chọn cài bundled Python và cho phép installer thêm Google Cloud CLI vào
`PATH`. Sau khi cài xong, đóng toàn bộ PowerShell/terminal trong VS Code rồi mở
terminal mới.

Kiểm tra cài đặt trong PowerShell:

```powershell
gcloud --version
Get-Command gcloud
ssh -V
```

Nếu `gcloud --version` vẫn không chạy, kiểm tra file cài đặt mặc định:

```powershell
Test-Path "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
```

Nếu kết quả là `True`, có thể dùng trực tiếp file này trong terminal hiện tại:

```powershell
& "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" --version
```

Sau đó mở terminal mới để nhận cấu hình `PATH` do installer thiết lập.

Nếu VS Code đã mở từ trước khi cài và terminal mới vẫn chưa nhận ra `gcloud`,
nạp SDK vào `PATH` của terminal hiện tại:

```powershell
$sdkBin = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin"
$env:Path = "$sdkBin;$env:Path"
gcloud --version
```

Thay đổi trên chỉ áp dụng cho terminal hiện tại. Đóng hoàn toàn rồi mở lại VS
Code để các terminal mới tự nhận User `PATH` đã được installer cấu hình.

## 3. Authenticate Google Cloud

Đăng nhập bằng Google account đã được cấp quyền vào project:

```powershell
gcloud auth login
gcloud config set project project-0c6ab625-3efb-4fe3-934
gcloud auth list
```

Nếu máy có nhiều Google account, chọn đúng account đã được cấp quyền:

```powershell
gcloud config set account YOUR_EMAIL
```

Lệnh `gcloud auth list` đánh dấu account đang hoạt động bằng dấu `*`.

## 4. Connect to the VM via IAP

### Windows PowerShell

Lần kết nối đầu tiên, `gcloud` có thể yêu cầu tạo SSH key và đăng ký public key
với Google Cloud:

```powershell
gcloud compute ssh rav-21-nuscenes-t4-spot `
  --project=project-0c6ab625-3efb-4fe3-934 `
  --zone=asia-southeast1-b `
  --tunnel-through-iap
```

Hoặc dùng một dòng để tránh lỗi khi copy ký tự xuống dòng:

```powershell
gcloud compute ssh rav-21-nuscenes-t4-spot --project=project-0c6ab625-3efb-4fe3-934 --zone=asia-southeast1-b --tunnel-through-iap
```

Trong PowerShell, không thêm `\` trước các tham số `--project`, `--zone` hoặc
`--tunnel-through-iap`. Dấu backtick `` ` `` chỉ dùng ở cuối dòng khi muốn tách
một lệnh PowerShell thành nhiều dòng.

### macOS/Linux

```bash
gcloud compute ssh rav-21-nuscenes-t4-spot \
  --project=project-0c6ab625-3efb-4fe3-934 \
  --zone=asia-southeast1-b \
  --tunnel-through-iap
```

Sau khi kết nối thành công, xác định Linux username và kiểm tra mount bằng các
lệnh chỉ đọc:

```bash
whoami
ls -la /data
df -h /data
```

Khi prompt đổi từ PowerShell, ví dụ `PS C:\...>`, thành dạng
`MSI@rav-21-nuscenes-t4-spot:~$` và Ubuntu hiển thị thông báo chào mừng, bạn đã
ở bên trong VM. Không chạy lại `gcloud compute ssh` tại prompt Linux này; lệnh
đó chỉ được chạy từ máy cá nhân. Dùng `exit` để ngắt kết nối và quay lại
PowerShell.

Mỗi thành viên dùng Linux username được tạo cho account của mình. Không sao
chép username từ cấu hình SSH của thành viên khác.

## 5. Connect with VS Code Remote - SSH

Trước tiên, kết nối một lần bằng `gcloud compute ssh` như ở phần trên để bảo
đảm SSH key đã được tạo.

### Windows

Tìm thư mục cài Google Cloud SDK:

```powershell
gcloud info --format="value(installation.sdk_root)"
```

Mở `%USERPROFILE%\.ssh\config`, sau đó thêm cấu hình dưới đây. Thay
`<LINUX_USER>`, `<WINDOWS_USER>` và đường dẫn SDK nếu kết quả của lệnh trên
khác với đường dẫn mẫu.

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

Kiểm tra alias bằng PowerShell:

```powershell
ssh rav21-t4 "whoami; ls -la /data"
```

Sau đó mở Command Palette trong VS Code và chọn **Remote-SSH: Connect to
Host...** → `rav21-t4`. Có thể mở trực tiếp `/data` bằng lệnh:

```powershell
code --remote ssh-remote+rav21-t4 /data
```

### macOS/Linux

Tìm đường dẫn `gcloud`:

```bash
which gcloud
```

Thêm cấu hình sau vào `~/.ssh/config`. Thay `<LINUX_USER>` và
`/absolute/path/to/gcloud` bằng giá trị tương ứng trên máy:

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

Kiểm tra alias và mở `/data`:

```bash
ssh rav21-t4 'whoami; ls -la /data'
code --remote ssh-remote+rav21-t4 /data
```

## 6. Data Layout

| Path | Nội dung | Quy tắc sử dụng |
| --- | --- | --- |
| `/data/nuscenes/dataset` | nuScenes trainval gốc | Chỉ đọc; có thể chứa ảnh chưa làm mờ |
| `/data/nuscenes/models` | Model EgoBlur | Không sửa hoặc thay file |
| `/data/nuscenes/logs` | Log do pipeline ghi | Chỉ dùng để kiểm tra tiến độ |
| `/data/nuscenes/work` | File tạm riêng của pipeline | Không truy cập hoặc sao chép |
| `/data/rav21-data/derived/previews/trainval` | Video preview đã xử lý | Dùng để review kết quả |
| `/data/rav21-data/scenes/trainval/<scene>` | Manifest, quality, privacy và sync report | Đọc báo cáo theo scene |
| `/data/rav21-data/catalog` | Catalog đầu ra | Chỉ đọc |
| `/data/<user-folder>` | Dữ liệu mới của thành viên | Chỉ ghi vào thư mục do chính thành viên tạo |
| `/opt/rav21` | Source code đang chạy trên VM | Không sửa trực tiếp khi production chạy |

Một số artifacts do production service tạo có thể chỉ cho owner đọc. Nếu gặp
`Permission denied`, gửi chính xác đường dẫn cho người quản trị. Không dùng
`sudo`, `chmod`, `chown` hoặc sao chép dữ liệu sang vị trí khác để né quyền.

## 7. Safe Read-only Checks

Kiểm tra trạng thái VM từ máy cá nhân:

```powershell
gcloud compute instances describe rav-21-nuscenes-t4-spot `
  --project=project-0c6ab625-3efb-4fe3-934 `
  --zone=asia-southeast1-b `
  --format="value(status)"
```

Sau khi SSH vào VM, có thể dùng các lệnh chỉ đọc sau:

```bash
df -h /data
ls -la /data
find /data/nuscenes/dataset -maxdepth 2 -type d | head
find /data/rav21-data/scenes/trainval -mindepth 1 -maxdepth 1 -type d | wc -l
tail -n 50 /data/nuscenes/logs/trainval-service.log
```

Không stop/start service, chạy lại pipeline, sửa source production, hoặc thay
đổi output production. Production dùng lock `/data/rav21-data/.trainval.lock`
để ngăn hai tiến trình cùng ghi.

## 8. Data Safety Rules

- Raw nuScenes có thể chứa khuôn mặt và biển số chưa làm mờ. Chỉ sử dụng trong
  phạm vi dự án và không chia sẻ ra ngoài.
- Không commit dataset, model, log, video hoặc credentials vào Git.
- Không ghi vào `/data/nuscenes` hoặc `/data/rav21-data`.
- Không xóa, di chuyển, đổi owner hoặc đổi permission của dữ liệu trên `/data`.
- Không tải toàn bộ dataset về máy cá nhân nếu chưa được thống nhất.
- Ưu tiên preview đã ẩn danh và các report JSON khi review dữ liệu.
- Khi cần output riêng, dùng thư mục đã được người vận hành cấp hoặc thư mục
  riêng của thành viên; không dùng output production.
- Nếu chưa chắc một thao tác có ghi dữ liệu hoặc ảnh hưởng production hay
  không, dừng lại và hỏi người đang vận hành.

## 9. Spot VM and Recovery

`rav-21-nuscenes-t4-spot` dùng NVIDIA T4 Spot GPU, vì vậy Google có thể preempt
VM khi thiếu capacity. Persistent disk `rav-21-nuscenes-data` vẫn là nơi lưu dữ
liệu tại `/data`; không lưu dữ liệu cần giữ lâu trên filesystem tạm của VM.

Watchdog thử start lại VM mỗi 5 phút. Nếu zone chưa có T4, các lần thử có thể
tiếp tục thất bại cho tới khi capacity quay lại. Khi VM boot thành công,
pipeline production resume từ checkpoint.

Thành viên chỉ kiểm tra trạng thái bằng lệnh ở phần 7. Không tự start/stop VM
hoặc production service.

## 10. Troubleshooting

### `gcloud` is not recognized

Google Cloud CLI chưa được cài hoặc chưa có trong `PATH`. Làm theo phần
**Cài Google Cloud CLI trên Windows**, sau đó đóng và mở lại PowerShell. Chỉ
chạy lệnh SSH khi cả hai lệnh sau trả về thành công:

```powershell
gcloud --version
Get-Command gcloud
```

Nếu file `gcloud.cmd` đã tồn tại nhưng VS Code vẫn báo lỗi, terminal đang dùng
`PATH` cũ. Nạp lại `PATH` bằng đoạn PowerShell ở phần cài đặt hoặc đóng hoàn
toàn rồi mở lại VS Code.

### `403` hoặc thiếu `iap.tunnelInstances.accessViaIAP`

Kiểm tra account và project đang hoạt động:

```powershell
gcloud auth list
gcloud config get-value account
gcloud config get-value project
```

Nếu đang dùng nhầm account, đăng nhập bằng Google account đã được cấp quyền và
kiểm tra account vừa đăng nhập:

```powershell
gcloud auth login
gcloud auth list
gcloud config set project project-0c6ab625-3efb-4fe3-934
```

Nếu có nhiều account, nhập email thật của account đã được cấp quyền khi
PowerShell yêu cầu:

```powershell
$authorizedAccount = Read-Host "Google account đã được cấp quyền"
gcloud config set account $authorizedAccount
```

Không nhập nguyên các chuỗi ví dụ hoặc placeholder như
`YOUR_AUTHORIZED_EMAIL`; chúng không có credentials.

Nếu account và project đều đúng nhưng lỗi báo thiếu `compute.instances.get`,
gửi nguyên thông báo lỗi cho quản trị viên. Account cần một role tối thiểu chứa
permission này (thường là **Compute Viewer**) và quyền IAP phù hợp (thường là
**IAP-secured Tunnel User**). Không tự thay đổi IAM.

### `compute.instances.get` sau khi đã thấy prompt của VM

Nếu prompt đã là `USER@rav-21-nuscenes-t4-spot:~$`, kết nối SSH ban đầu đã
thành công. Lỗi này thường xuất hiện khi chạy lại `gcloud compute ssh` từ bên
trong VM, nơi identity của VM không có quyền đọc Compute Engine instance. Không
thay đổi IAM và không tạo kết nối lồng nhau. Tiếp tục làm việc bằng các lệnh
Linux cần thiết, hoặc chạy `exit` rồi thực hiện lệnh `gcloud compute ssh` từ
PowerShell trên máy cá nhân.

### `Connection timed out` hoặc VS Code vẫn trỏ tới IP cũ

VM không có public IP. Dùng alias `rav21-t4` với `ProxyCommand` IAP ở trên và
xóa tham chiếu tới IP cũ khỏi cấu hình local nếu có.

### VM có trạng thái `TERMINATED`

Đây có thể là Spot preemption. Watchdog kiểm tra mỗi 5 phút và thử start lại VM
trong cùng zone. Không tự start VM; báo cho người vận hành nếu trạng thái kéo
dài hoặc cần xử lý khẩn cấp.

### Cảnh báo tăng tốc NumPy khi mở IAP tunnel

Cảnh báo NumPy chỉ ảnh hưởng tốc độ truyền của IAP tunnel; nó không làm sai dữ
liệu và không ngăn SSH hoạt động.

### `Permission denied` bên trong `/data`

Không tự đổi owner hoặc permission. Gửi đường dẫn bị lỗi cho quản trị viên để
được cấp quyền tối thiểu cần thiết.

### SSH key hoặc username không đúng

Kết nối lại bằng `gcloud compute ssh` trước, sau đó chạy `whoami`. Cập nhật
`User` và `IdentityFile` trong SSH config bằng đúng username và key của account
cá nhân.
