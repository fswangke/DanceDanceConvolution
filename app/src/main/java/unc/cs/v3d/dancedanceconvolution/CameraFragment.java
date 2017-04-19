package unc.cs.v3d.dancedanceconvolution;

import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.IntDef;
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
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Created by kewang on 4/18/17.
 */

public class CameraFragment extends Fragment {
    private static final String TAG = CameraFragment.class.getSimpleName();
    private static final int MINIMUM_PREVIEW_SIZE = 320;
    private static final int CPM_PREVIEW_WIDTH = 256;
    private static final int CPM_PREVIEW_HEIGHT = 256;

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    private static final String FRAGMENT_DIALOG = "dialog";

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private final TextureView.SurfaceTextureListener mSurfaceTextureListener = new TextureView.SurfaceTextureListener() {
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
     * Callback for Activities to use to initialize their data once the
     * selected preview size is known.
     */
    public interface ConnectionCallback {
        void onPreviewSizeChosen(Size size, int cameraRotation);
    }

    private ConnectionCallback mConnectionCallback;

    /**
     * ID for the current camera.
     */
    private String mCameraId;

    /**
     * TextureView for camera preview.
     */
    private AutoFitTextureView mTextureView;

    /**
     * CameraCaptureSession for camera preview;
     */
    private CameraCaptureSession mCaptureSession;

    /**
     * A reference to the opened CameraDevice.
     */
    private CameraDevice mCameraDevice;

    /**
     * The rotation in degrees of the camera sensor from the display.
     */
    private Integer mSensorOrientation;

    /**
     * android.util.Size of camera preview.
     */
    private Size mPreviewSize;

    /**
     * android.hardware.camera2.CameraDevice.StateCallback is called
     * when CameraDevice changes its state.
     */
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
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

    /**
     * An additional thread for running tasks that shouldn't block the UI.
     */
    private HandlerThread mBackgroundThread;

    /**
     * A Handler for running tasks in the background thread.
     */
    private Handler mBackgroundHandler;

    /**
     * An ImageReader that handles preview from camera.
     */
    private ImageReader mPreviewReader;

    /**
     * android.hardware.camera2.CaptureRequest.Builder: CameraRequest.Builder
     * for the camera preview
     */
    private CaptureRequest.Builder mCaptureRequestBuilder;

    /**
     * CaptureRequest generated by the request builder
     */
    private CaptureRequest mCaptureRequest;

    /**
     * A Semaphore to prevent the app from existing before closing the camera.
     */
    private final Semaphore mCameraOpenCloseLock = new Semaphore(1);

    /**
     * An OnImageAvailableListener to receive frames as they become available.
     */
    private final ImageReader.OnImageAvailableListener mImageListener;

    /**
     * Input size in pixels desired by CPM model.
     */
    private final Size mCPMPersonNetInputSize;
    private final Size mCPMPoseNetInputSize;

    /**
     * Layout identifier to inflate this Fragment.
     */
    private final int mLayout;

    // TODO: fragment contor

    /**
     * Show a Toast on the UI thread.
     */
    private void showToast(final String text) {
        final Activity activity = getActivity();
        if (activity != null) {
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(activity, text, Toast.LENGTH_SHORT).show();
                    ;
                }
            });
        }
    }

    /**
     * Given choices of sizes supported by a camera, chooses the smallest one whose width
     * and height are at least at the minimum of both, or an exact match if possible.
     */
    private static Size chooseOptimalSize(final Size[] choices, final int width, final int height) {
        // TODO(fswangke): fix the CPM size choice problem
        final int minSize = Math.max(Math.min(width, height), MINIMUM_PREVIEW_SIZE);
        final Size desiredSize = new Size(width, height);

        boolean exactSizeFound = false;
        final List<Size> bigEnough = new ArrayList<>();
        final List<Size> tooSmall = new ArrayList<>();
        for (final Size option : choices) {
            if (option.equals(desiredSize)) {
                exactSizeFound = true;
            }

            if (option.getHeight() >= minSize && option.getWidth() >= minSize) {
                bigEnough.add(option);
            } else {
                tooSmall.add(option);
            }
        }

        Log.i(TAG, "Desired size:" + desiredSize + ", min size: " + minSize + "x" + minSize);
        Log.i(TAG, "Valid preview sizes: [" + TextUtils.join(", ", bigEnough) + "]");
        Log.i(TAG, "Rejected preview sizes: [" + TextUtils.join(", ", tooSmall) + "]");

        if (exactSizeFound) {
            Log.i(TAG, "Exact size match found.");
            return desiredSize;
        }
        if (bigEnough.size() > 0) {
            final Size chosenSize = Collections.min(bigEnough, new CompareSizesByArea());
            Log.i(TAG, "Chosen size: " + chosenSize);
            return chosenSize;
        } else {
            Log.e(TAG, "Count not find any suitable preview size.");
            return choices[0];
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size o1, Size o2) {
            return Long.signum((long) o1.getWidth() * o1.getHeight() - (long) o2.getWidth() * o2.getHeight());
        }
    }

    public static CameraFragment newInstance() {
        // TODO: (fswangke) fix the default ctor
        return new CameraFragment();
    }

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(mLayout, container, false);
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

        // When the screen is turned of and turned back on, the SurfaceTexture is already available
        // "onSurfaceTextureAvailable" won't be called.
        // In this case, we open up a camera and start preview here, otherwise we wait until the
        // surface is ready in the SurfaceTextureListener
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

    /**
     * Sets up member variables related to camera.
     */
    private void setUpCameraOutputs(final int width, final int height) {
    }

    /**
     * Opens the camera specified by cameraId
     */
    private void openCamera(final int width, final int height) {
        setUpCameraOutputs(width, height);
        configureTransform(width, height);

        final Activity activity = getActivity();
        final CameraManager cameraManager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);

        try {
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            cameraManager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (final CameraAccessException e) {
            Log.e(TAG, "Exception!");
        } catch (final InterruptedException e) {
            throw new RuntimeException("Interrupted when trying to lock camera openining", e);
        } catch (final SecurityException se) {
            throw new SecurityException("Camera permission not granted.");
        }
    }

    /**
     * Close the camera specified by mCameraId
     */
    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            if (null != mCaptureSession) {
                mCaptureSession.close();
                mCaptureSession = null;
            }
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            if (null != mPreviewReader) {
                mPreviewReader.close();
                mPreviewReader = null;
            }
        } catch (final InterruptedException e) {
            throw new RuntimeException(e);
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    /**
     * Starts a background thread and its handler
     */
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("ImageListener");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /**
     * Stops the background thread and its handler
     */
    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "Exception");
        }
    }

    private final CameraCaptureSession.CaptureCallback mCaptureCallbak =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureProgressed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureResult partialResult) {
                }

                @Override
                public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
                }
            };

    /**
     * Create a new CameraCaptureSession for camera preview.
     */
    private void createCameraPreviewSession() {
        try {
            final SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());

            // configure the size of default buffer to be the size of camera preview
            final Surface surface = new Surface(texture);

            mCaptureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mCaptureRequestBuilder.addTarget(surface);

            Log.i(TAG, "Opening camera preview: " + mPreviewSize.getWidth() + "x" + mPreviewSize.getHeight());

            mPreviewReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(), ImageFormat.YUV_420_888, 2);
            mPreviewReader.setOnImageAvailableListener(mImageListener, mBackgroundHandler);
            mCaptureRequestBuilder.addTarget(mPreviewReader.getSurface());

            mCameraDevice.createCaptureSession(
                    Arrays.asList(surface, mPreviewReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            if (null == mCameraDevice) {
                                return;
                            }

                            // when the session is ready, we start displaying the preview
                            mCaptureSession = session;
                            try {
                                // auto focus
                                mCaptureRequestBuilder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                                // enable flash when necessary
                                mCaptureRequestBuilder.set(
                                        CaptureRequest.CONTROL_AE_MODE,
                                        CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                                // start displaying the camera preview
                                mCaptureRequest = mCaptureRequestBuilder.build();
                                mCaptureSession.setRepeatingRequest(
                                        mCaptureRequest,
                                        mCaptureCallbak,
                                        mBackgroundHandler
                                );
                            } catch (final CameraAccessException e) {
                                Log.d(TAG, "Exception!");
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            showToast("Failed");

                        }
                    }, null
            );
        } catch (final CameraAccessException e) {
            Log.e(TAG, "Exception!");
        }
    }

    /**
     * Configures the necessary transformation to mTextureView.
     * This method should be called after the camera preview size is determined.
     */
    private void configureTransform(final int viewWidth, final int viewHeight) {
        final Activity activity = getActivity();
        if (null == mTextureView || null == mPreviewSize || null == activity) {
            return;
        }
        final int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        final Matrix matrix = new Matrix();
        final RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        final RectF bufferRect = new RectF(0, 0, mPreviewSize.getWidth(), mPreviewSize.getHeight());
        final float centerX = viewRect.centerX();
        final float centerY = viewRect.centerY();

        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            final float scale =
                    Math.max((float) viewHeight / (float) mPreviewSize.getHeight(),
                            (float) viewWidth / (float) mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }
}

