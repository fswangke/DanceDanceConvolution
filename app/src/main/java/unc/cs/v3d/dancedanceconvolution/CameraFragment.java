package unc.cs.v3d.dancedanceconvolution;

import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.text.TextUtils;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

// Use camera 2 api to output camera real-time preview to two surface locations:
// 1) the SurfaceTexture for preview
// 2) the ImageReader for Tensorflow inference
// Once the Camera is initialized with a set-up preview size, it calls the OnCameraConnectionCallback callback
// to tell the main activity about the size and rotation of the
public class CameraFragment extends Fragment {
    private static final String TAG = CameraFragment.class.getSimpleName();

    public interface OnCameraConnectionCallback {
        void onPreviewSizeChosen(Size size, int rotation);
    }

    private final TextureView.SurfaceTextureListener mSurfaceTextureListener =
            new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                    openCamera(width, height);
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
                    configureTransform(width, height);
                }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                    return false;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture surface) {

                }
            };

    /**
     * The ID of the camera. In this app we only use rear-facing cameras.
     */
    private String mCameraId;

    /**
     * AutoFitTextureView for camera view.
     */
    private AutoFitTextureView mTextureView;

    /**
     * CameraCaptureSession for capture camera preview.
     */
    private CameraCaptureSession mCameraCaptureSession;

    /**
     * Reference to opened camera device.
     */
    private CameraDevice mCameraDevice;

    /**
     * Rotation in degrees of the camera sensor from the display.
     */
    private Integer mSensorRotation;

    /**
     * Assuming a up-right screen orientation (0) and convert screen rotation to JPEG orientation.
     * (sensorOrientation - 0 + 360) % 360
     */
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    /**
     * Size of camera previw
     */
    private Size mPreviewSize;

    /**
     * CameraDevice.StateCallback is called when CameraDevice changes its state.
     */
    private final CameraDevice.StateCallback mCameraDeviceStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            // This method is called when the camera is opened.
            // Start camera preview here.
            mCameraOpenCloseLock.release();
            mCameraDevice = camera;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            mCameraOpenCloseLock.release();
            camera.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            mCameraOpenCloseLock.release();
            camera.close();
            mCameraDevice = null;
            final Activity activity = getActivity();
            if (null != activity) {
                activity.finish();
            }
        }
    };

    private CameraCaptureSession.StateCallback mCameraCaptureSessionStateCallback =
            new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    if (null == mCameraDevice) {
                        return;
                    }

                    mCameraCaptureSession = session;
                    try {
                        // Set auto focus for preview
                        mPreviewRequestBuilder.set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                        // Set flash
                        mPreviewRequestBuilder.set(
                                CaptureRequest.CONTROL_AE_MODE,
                                CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                        // start display camera preview
                        mPreviewRequest = mPreviewRequestBuilder.build();
                        mCameraCaptureSession.setRepeatingRequest(mPreviewRequest, mCaptureCallback, mBackgroundHandler);
                    } catch (final CameraAccessException cae) {
                        Log.e(TAG, "Exception.");
                    }
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                    Log.wtf(TAG, "Failed");

                }

                @Override
                public void onReady(@NonNull CameraCaptureSession session) {
                    super.onReady(session);
                }

                @Override
                public void onActive(@NonNull CameraCaptureSession session) {
                    super.onActive(session);
                }

                @Override
                public void onClosed(@NonNull CameraCaptureSession session) {
                    super.onClosed(session);
                }

                @Override
                public void onSurfacePrepared(@NonNull CameraCaptureSession session, @NonNull Surface surface) {
                    super.onSurfacePrepared(session, surface);
                }
            };

    /**
     * Background thread and handler
     */
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;

    /**
     * ImageReader that receives previews from capture.
     * Parent activity get the image pixel data using this ImageReader and feed into Tensorflow
     * for inference.
     */
    private ImageReader mPreviewImageReader;

    /**
     * camera capture request builder
     */
    private CaptureRequest.Builder mPreviewRequestBuilder;

    /**
     * camera capture request
     */
    private CaptureRequest mPreviewRequest;

    /**
     * Semaphore to prevent the app from exiting before closing the camera.
     */
    private final Semaphore mCameraOpenCloseLock = new Semaphore(1);

    /**
     * OnImageAvailableListener callback.
     * Use the callback to send pixel data to tensorflow for inference.
     */
    private ImageReader.OnImageAvailableListener mImageAvailableListener;

    /**
     * Input size in pixels desired by Tensorflow.
     */
    private final Size mInputSize;

    /**
     * CameraConnectionCallback.
     * When camera is set-up with preview. Call this to setup tensor-flow preprocessing.
     * YUV->RGB conversion, etc.
     */
    private final OnCameraConnectionCallback mCameraConnectionCallback;

    /**
     * Layout container R.id
     */
    private int mLayoutId;

    /**
     * Singleton pattern.
     */
    private CameraFragment(
            final OnCameraConnectionCallback onCameraConnectionCallback,
            final ImageReader.OnImageAvailableListener onImageAvailableListener,
            final int layout,
            final Size inputSize) {
        this.mCameraConnectionCallback = onCameraConnectionCallback;
        this.mImageAvailableListener = onImageAvailableListener;
        this.mLayoutId = layout;
        this.mInputSize = inputSize;
    }

    public static CameraFragment getInstance(
            final OnCameraConnectionCallback onCameraConnectionCallback,
            final ImageReader.OnImageAvailableListener onImageAvailableListener,
            final int layout,
            final Size inputSize) {
        return new CameraFragment(onCameraConnectionCallback, onImageAvailableListener, layout, inputSize);
    }

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(mLayoutId, container, false);
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        mTextureView = (AutoFitTextureView) view.findViewById(R.id.texture_view);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();

        if (mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("ImageListener");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "Exception", e);
        }
    }

    /**
     * Compares two Sizes by area.
     */
    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size o1, Size o2) {
            return Long.signum(
                    (long) o1.getWidth() * o1.getHeight() -
                            (long) o2.getWidth() * o2.getHeight()
            );
        }
    }

    /**
     * Given choices of sizes supported by camera hardware, chooses the smallest one
     * whose width and height are at least as large as the minimum of both,
     * or exact match if possible.
     */
    private final static int MINIMUM_PREVIEW_SIZE = 320;

    private static Size chooseOptimalSize(final Size[] choices, final int width, final int height) {
        final int minSize = Math.max(Math.min(width, height), MINIMUM_PREVIEW_SIZE);
        final Size desiredSize = new Size(width, height);

        boolean exactSizeFound = false;
        final List<Size> bigEnough = new ArrayList<>();
        final List<Size> tooSmall = new ArrayList<>();
        for (final Size choice : choices) {
            if (choice.equals(desiredSize)) {
                exactSizeFound = true;
            }

            if (choice.getWidth() >= desiredSize.getWidth() && choice.getHeight() >= choice.getHeight()) {
                bigEnough.add(choice);
            } else {
                tooSmall.add(choice);
            }
        }

        Log.i(TAG, "Desired size: " + desiredSize);
        Log.i(TAG, "Valid preview sizes: " + TextUtils.join(", ", bigEnough));
        Log.i(TAG, "Rejected preview sizes: " + TextUtils.join(", ", tooSmall));

        if (exactSizeFound) {
            Log.i(TAG, "Exact size match found.");
            return desiredSize;
        }

        if (bigEnough.size() > 0) {
            final Size chosenSize = Collections.min(bigEnough, new CompareSizesByArea());
            Log.i(TAG, "Chosen size: " + chosenSize);
            return chosenSize;
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size.");
            return choices[0];
        }
    }

    /**
     * Sets up member variables related to camera.
     */
    private void setUpCameraOutputs(final int width, final int height) {
        final Activity activity = getActivity();
        final CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics cc = manager.getCameraCharacteristics(cameraId);
                final Integer facing = cc.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    Log.i(TAG, "Skipping facing camera");
                    continue;
                }

                final StreamConfigurationMap map = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    Log.i(TAG, "StreamConfigurationMap not available for Camera: " + cameraId);
                    continue;
                }

                mSensorRotation = cc.get(CameraCharacteristics.SENSOR_ORIENTATION);
                mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class), mInputSize.getWidth(), mInputSize.getHeight());

                final int orientation = getResources().getConfiguration().orientation;
                if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                    mTextureView.setAspectRatio(mPreviewSize.getWidth(), mPreviewSize.getHeight());
                } else {
                    mTextureView.setAspectRatio(mPreviewSize.getHeight(), mPreviewSize.getWidth());
                }
                mCameraId = cameraId;
            }
            if (mCameraId == null) {
                throw new Exception("No supported camera found!");
            }
        } catch (final CameraAccessException cae) {
            Log.e(TAG, "CameraAccessException", cae);
        } catch (final NullPointerException npe) {
            Log.e(TAG, "NullPointerException", npe);
        } catch (final Exception e) {
            Log.e(TAG, e.getMessage());
        }

        Log.e(TAG, "Ever here?");
        mCameraConnectionCallback.onPreviewSizeChosen(mPreviewSize, mSensorRotation);
        Log.e(TAG, "Really here.");
    }

    /**
     * Configures the necessary Matrix transformation to `mTextureView`.
     * This method should be called after the camera preview size is determined.
     *
     * @param viewWidth  The width of the `mTextureView`
     * @param viewHeight The height of the `mTextureView`
     */
    private void configureTransform(final int viewWidth, final int viewHeight) {
        final Activity activity = getActivity();
        if (null == mTextureView || null == mPreviewSize || null == activity) {
            return;
        }

        final int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        final Matrix matrix = new Matrix();
        final RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        final RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        final float centerX = viewRect.centerX();
        final float centerY = viewRect.centerY();

        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            final float scale = Math.max((float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }

        mTextureView.setTransform(matrix);
    }

    private void openCamera(final int width, final int height) {
        setUpCameraOutputs(width, height);
        configureTransform(width, height);

        final Activity activity = getActivity();
        final CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out when locking the camera.");
            }
            manager.openCamera(mCameraId, mCameraDeviceStateCallback, mBackgroundHandler);
        } catch (final CameraAccessException cae) {
            Log.e(TAG, "CameraAccessException", cae);
        } catch (final InterruptedException e) {
            throw new RuntimeException("Interrupted.", e);
        }
    }

    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            if (null != mCameraCaptureSession) {
                mCameraCaptureSession.close();
                mCameraCaptureSession = null;
            }
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            if (null != mPreviewImageReader) {
                mPreviewImageReader.close();
                mPreviewImageReader = null;
            }
        } catch (final InterruptedException e) {
            throw new RuntimeException("Interrupted.", e);
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    private final CameraCaptureSession.CaptureCallback mCaptureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureStarted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, long timestamp, long frameNumber) {
                    super.onCaptureStarted(session, request, timestamp, frameNumber);
                }

                @Override
                public void onCaptureProgressed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureResult partialResult) {
                    super.onCaptureProgressed(session, request, partialResult);
                }

                @Override
                public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                }

                @Override
                public void onCaptureFailed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureFailure failure) {
                    super.onCaptureFailed(session, request, failure);
                }

                @Override
                public void onCaptureSequenceCompleted(@NonNull CameraCaptureSession session, int sequenceId, long frameNumber) {
                    super.onCaptureSequenceCompleted(session, sequenceId, frameNumber);
                }

                @Override
                public void onCaptureSequenceAborted(@NonNull CameraCaptureSession session, int sequenceId) {
                    super.onCaptureSequenceAborted(session, sequenceId);
                }

                @Override
                public void onCaptureBufferLost(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull Surface target, long frameNumber) {
                    super.onCaptureBufferLost(session, request, target, frameNumber);
                }
            };

    private void createCameraPreviewSession() {
        try {
            final SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            final Surface surface = new Surface(texture);

            mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);

            mPreviewImageReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(), ImageFormat.YUV_420_888, 2);
            mPreviewImageReader.setOnImageAvailableListener(mImageAvailableListener, mBackgroundHandler);
            mPreviewRequestBuilder.addTarget(mPreviewImageReader.getSurface());

            mCameraDevice.createCaptureSession(Arrays.asList(surface, mPreviewImageReader.getSurface()),
                    mCameraCaptureSessionStateCallback, null);

        } catch (final CameraAccessException cae) {
            Log.e(TAG, "Exception:", cae);
        }
    }
}
