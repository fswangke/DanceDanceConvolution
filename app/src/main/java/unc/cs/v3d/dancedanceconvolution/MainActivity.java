package unc.cs.v3d.dancedanceconvolution;

import android.Manifest;
import android.app.Fragment;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.os.Trace;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.KeyEvent;
import android.view.WindowManager;
import android.widget.Toast;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Vector;

// Convolutional Pose Machines (CPM) have two stages: 1) PersonNetwork 2) PoseNet
// TODO: (fswangke) fire-up PersonNetwork
// TODO: (fswangke) convert PersonNet output to PoseNet input via C++ library
// TODO: (fswangke) fire-up PoseNetwork
// TODO: (fswangke) render the pose back to the image to visualize the Pose

public class MainActivity extends AppCompatActivity implements
        ImageReader.OnImageAvailableListener,
        CameraFragment.OnCameraConnectionCallback,
        ActivityCompat.OnRequestPermissionsResultCallback {
    static {
        try {
            System.loadLibrary("colorspace_conversion");
            System.loadLibrary("tensorflow_inference");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static final int PERMISSIONS_REQUEST = 123;
    private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;
    private static final String STORAGE_PERMISSION = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private HandlerThread mTFInferenceThread;
    private Handler mTFInferenceHandler;
    private Bitmap mCroppedBitmap = null;
    private Bitmap mRgbFrameBitmap = null;
    private BorderedText mBorderedText;
    private Matrix mCropToFrameTransform;
    private Matrix mFrameToCropTransform;
    private OverlayView mOverlayView;
    private PoseMachine mPoseMachine;
    private boolean mIsInferring = false;
    private boolean mShowTFRuntimeStats = false;
    private byte[][] mYuvBuffer;
    private int mPreviewHeight = 0;
    private int mPreviewWidth = 0;
    private int mSensorOrientation;
    private int[] mRgbBuffer;
    private long mLastInferenceTime;
    private static final String INCEPTION_MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String PERSON_MODEL_FILE = "file:///android_asset/person_net.pb";
    private static final String POSE_MODEL_FILE = "";
    private static final String TAG = MainActivity.class.getSimpleName();
    private static final float TEXT_SIZE_DIP = 10;
    static final private String INCEPTION_INPUT_NODE_NAME = "input";
    static final private String INCEPTION_OUTPUT_NODE_NAME = "output";
    static final private int INCEPTION_IMAGE_MEAN = 117;
    static final private int INCEPTION_IMAGE_STD = 1;
    static final private int INCEPTION_INPUT_SIZE = 224;
    static final private int PERSON_NET_HEIGHT = 376;
    static final private int PERSON_NET_WIDTH = 656;
    static final private boolean USE_INCEPTION = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }

//        mPoseMachine = PoseMachine.getPoseMachine(getAssets(),
//                PERSON_MODEL_FILE,
//                INCEPTION_INPUT_SIZE,
//                INCEPTION_IMAGE_MEAN,
//                INCEPTION_IMAGE_STD,
//                INCEPTION_INPUT_NODE_NAME,
//                INCEPTION_OUTPUT_NODE_NAME);
        mPoseMachine = PoseMachine.getPoseMachine(getAssets(),
                PERSON_MODEL_FILE,
                PERSON_NET_WIDTH,
                PERSON_NET_HEIGHT,
                "image",
                "conv1_1");
    }

    @Override
    protected void onStart() {
        Log.d(TAG, "onStart");
        super.onStart();
    }

    @Override
    protected void onResume() {
        Log.d(TAG, "onResume");
        super.onResume();

        startTFInferenceThread();
    }

    @Override
    protected void onPause() {
        Log.d(TAG, "onPause");
        stopTFInferenceThread();
        super.onPause();
    }

    @Override
    protected void onStop() {
        Log.d(TAG, "onStop");
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        Log.d(TAG, "onDestroy");
        super.onDestroy();
    }


    @Override
    public void onImageAvailable(ImageReader reader) {
        Image image = null;
        image = reader.acquireLatestImage();
        if (image == null) {
            return;
        }

        if (mIsInferring) {
            image.close();
            return;
        }

        mIsInferring = true;
        Trace.beginSection("TensorFlowInference");

        final Image.Plane[] planes = image.getPlanes();
        fillBuffer(planes, mYuvBuffer);
        final int yRowStride = planes[0].getRowStride();
        final int uvRowStride = planes[1].getRowStride();
        final int uvPixelStride = planes[1].getPixelStride();
        ImageUtils.convertYUV420ToARGB8888(
                mYuvBuffer[0],
                mYuvBuffer[1],
                mYuvBuffer[2],
                mRgbBuffer,
                mPreviewWidth,
                mPreviewHeight,
                yRowStride,
                uvRowStride,
                uvPixelStride,
                false);
        image.close();

        mRgbFrameBitmap.setPixels(mRgbBuffer, 0, mPreviewWidth, 0, 0, mPreviewWidth, mPreviewHeight);
        final Canvas canvas = new Canvas(mCroppedBitmap);
        canvas.drawBitmap(mRgbFrameBitmap, mFrameToCropTransform, null);

        runInTFInferenceThread(new Runnable() {
            @Override
            public void run() {
                final long startTime = SystemClock.uptimeMillis();
                mPoseMachine.inferPose(mCroppedBitmap);
                mLastInferenceTime = SystemClock.uptimeMillis() - startTime;
                mIsInferring = false;
                if (mShowTFRuntimeStats) {
                    Log.v(TAG, "TIME (ms): " + mLastInferenceTime);
                    mOverlayView.postInvalidate();
                }
            }
        });
        Trace.endSection();
    }


    @Override
    public void onPreviewSizeChosen(Size size, int rotation) {
        final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP,
                TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        mBorderedText = new BorderedText(textSizePx);
        mBorderedText.setTypeFace(Typeface.MONOSPACE);


        mPreviewHeight = size.getHeight();
        mPreviewWidth = size.getWidth();

        mOverlayView = (OverlayView) findViewById(R.id.stats_overlay_view);
        mOverlayView.addCallback(new OverlayView.DrawCallback() {
            @Override
            public void drawCallback(Canvas canvas) {
                if (!mShowTFRuntimeStats) {
                    return;
                }

                final Vector<String> lines = new Vector<String>();


                final String statString = mPoseMachine.getStatString();
                String[] statLines = statString.split("\n");
                lines.addAll(Arrays.asList(statLines));

                lines.add("Frame: " + mPreviewWidth + "x" + mPreviewHeight);
                lines.add("Crop:  " + mCroppedBitmap.getWidth() + "x" + mCroppedBitmap.getHeight());
                lines.add("View:  " + canvas.getWidth() + "x" + canvas.getHeight());
                lines.add("Rotation: " + mSensorOrientation);
                lines.add("Inference time: " + mLastInferenceTime + "ms");


                mBorderedText.drawLiness(canvas, 10, canvas.getHeight() - 10, lines);
            }
        });

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();
        Log.d(TAG, "Sensor rotation " + rotation + " screen rotation " + screenOrientation);
        mSensorOrientation = screenOrientation + rotation;

        mRgbBuffer = new int[mPreviewWidth * mPreviewHeight];
        mYuvBuffer = new byte[3][];
        mRgbFrameBitmap = Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Bitmap.Config.ARGB_8888);
        mCroppedBitmap = Bitmap.createBitmap(PERSON_NET_WIDTH, PERSON_NET_HEIGHT,
                Bitmap.Config.ARGB_8888);

        mFrameToCropTransform = ImageUtils.getTransformationMatrix(mPreviewWidth, mPreviewHeight,
                PERSON_NET_WIDTH, PERSON_NET_HEIGHT, mSensorOrientation, false);
        mCropToFrameTransform = new Matrix();
        mFrameToCropTransform.invert(mCropToFrameTransform);
    }

    private void fillBuffer(final Image.Plane[] channels, final byte[][] yuvBuffer) {
        for (int i = 0; i < channels.length; ++i) {
            final ByteBuffer buffer = channels[i].getBuffer();
            if (yuvBuffer[i] == null) {
                yuvBuffer[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBuffer[i]);
        }
    }


    private void startTFInferenceThread() {
        mTFInferenceThread = new HandlerThread("TFInference");
        mTFInferenceThread.start();
        mTFInferenceHandler = new Handler(mTFInferenceThread.getLooper());
    }

    private void stopTFInferenceThread() {
        mTFInferenceThread.quitSafely();
        try {
            mTFInferenceThread.join();
            mTFInferenceThread = null;
            mTFInferenceHandler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "Exception", e);
        }
    }

    private void runInTFInferenceThread(final Runnable r) {
        if (mTFInferenceHandler != null) {
            mTFInferenceHandler.post(r);
        }
    }


    private boolean hasPermission() {
        return checkSelfPermission(CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(STORAGE_PERMISSION) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {
        if (shouldShowRequestPermissionRationale(CAMERA_PERMISSION) ||
                shouldShowRequestPermissionRationale(STORAGE_PERMISSION)) {
            Toast.makeText(MainActivity.this, "Camera and Storage are necessary for this app.",
                    Toast.LENGTH_SHORT).show();
        }
        requestPermissions(new String[]{CAMERA_PERMISSION, STORAGE_PERMISSION}, PERMISSIONS_REQUEST);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                        grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                    setFragment();
                } else {
                    requestPermission();
                }
            }
        }
    }

    private void setFragment() {
        final Fragment fragment = CameraFragment.getInstance(
                // fixme(fswangke): adjust fragment size based on CPM required input size
                this, this, R.layout.camera_fragment, new Size(720, 480));

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            Log.v(TAG, "Trying to trigger overlay display");
            mShowTFRuntimeStats = !mShowTFRuntimeStats;
            mPoseMachine.enableStatLogging(mShowTFRuntimeStats);
            mOverlayView.postInvalidate();
            return true;
        }

        return super.onKeyDown(keyCode, event);
    }
}
