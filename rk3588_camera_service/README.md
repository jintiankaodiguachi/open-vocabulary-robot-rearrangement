# RK3588 Camera Service

Final responsibility:

`HIK -> OpenCV/V4L2 -> BGR -> undistort -> UNDISTORTED_1280x720 -> JPEG/HTTP -> A6000`

Do not apply Homography on RK3588.

## Put calibration file here

Place `camera_intrinsics.npz` beside the scripts.

The public repository does not include calibration from the original robot.
Calibrate your own camera and provide this file before starting with
`camera.do_undistort: true`. The backend intentionally raises an error when
the calibration is missing. Set `camera.device` in a local config copy to
your V4L2 device; `0` is only an example.

Supported camera-matrix keys:
`K`, `camera_matrix`, `cameraMatrix`, `mtx`

Supported distortion keys:
`dist`, `dist_coeffs`, `distCoeffs`, `distortion_coefficients`

## Check keys

```bash
python3 - <<'PY'
import numpy as np
d=np.load('camera_intrinsics.npz')
print(d.files)
for k in d.files:
    print(k, d[k].shape)
PY
```

## Local test

```bash
python3 test_camera.py --config config.yaml
```

Expected:

```text
frame.shape = (720, 1280, 3)
```

and `test_undistorted.jpg` must be in exactly the same pixel coordinate space used to calibrate Homography.

## Start service

```bash
chmod +x run.sh
./run.sh
```

Use `./run.sh /absolute/path/to/config.local.yaml` or set `RK_CAMERA_CONFIG`
to select another config. Set `PYTHON` to select the Python interpreter.

## Test on RK3588

```bash
curl http://127.0.0.1:8080/health
wget -O scene.jpg http://127.0.0.1:8080/api/v1/frame/latest.jpg
```

## Test from A6000

```bash
curl http://camera-host.local:8080/health
wget -O scene.jpg http://camera-host.local:8080/api/v1/frame/latest.jpg
```

Resolve `camera-host.local` to your RK3588 address on your own network.
`GET /api/v1/frame/latest/meta` returns the frame ID, timestamp, resolution,
encoding, and image-space ID. The JPEG response includes the same metadata
in `X-Frame-Id`, `X-Timestamp`, and `X-Image-Space` headers.

## Critical rule

Default runtime undistortion is equivalent to using `K` as the output camera matrix.
There is no crop and no resize.

If your old Homography calibration used `cv2.getOptimalNewCameraMatrix()` / `newK`, configure `new_camera_matrix_key` to that saved matrix key.
If your old pipeline cropped the undistorted image, do not reuse the Homography until the runtime image pipeline exactly matches the calibration pipeline.
