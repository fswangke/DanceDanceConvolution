package unc.cs.v3d.dancedanceconvolution;

import android.Manifest;
import android.app.Fragment;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.os.Parcelable;
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
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

// Convolutional Pose Machines (CPM) have two stages: 1) PersonNetwork 2) PoseNet
// TODO: (fswangke) post-process the PafNet results via C++
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

    private static final String TAG = MainActivity.class.getSimpleName();

    // Permission handling
    private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;
    private static final String STORAGE_PERMISSION = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final int PERMISSIONS_REQUEST = 123;

    // Background threads to run Tensorflow inference
    private Handler mTFInferenceHandler;
    private HandlerThread mTFInferenceThread;

    // camera preview image and cropped image for tensorflow
    private Bitmap mCroppedBitmap = null;
    private Bitmap mRgbFrameBitmap = null;
    private Matrix mCropToFrameTransform;
    private Matrix mFrameToCropTransform;
    private byte[][] mYuvBuffer;
    private int mPreviewHeight = 0;
    private int mPreviewWidth = 0;
    private int mSensorOrientation;
    private int[] mRgbBuffer;

    // debug and running statstics info
    private BorderedText mBorderedText;
    private OverlayView mOverlayView;

    // tensorflow model and classifier
    private PoseMachine mPoseMachine;
    private boolean mIsInferring = false;
    private boolean mShowTFRuntimeStats = false;
    private long mLastInferenceTime;
    private static final float TEXT_SIZE_DIP = 10;
    private static final String PAFNET_MODEL_FILE = "file:///android_asset/paf_net.pb";
    static final private String PAFNET_INPUT_NODE_NAME = "image";
    static final private String[] PAFNET_OUTPUT_NODE_NAMES = new String[]{"conv5_5_CPM_L2"};
    static final private int PAFNET_INPUT_SIZE = 224;
    static final private int PAFNET_OUTPUT_SIZE = 28;

    static final private int NUM_INSTRUCTIONS = 5;
    private Button[] buttons_instruction_correct;
    private Button[] buttons_instruction_infer;

    private MediaPlayer mMediaPlayer = null;
    private final int timeStep = 1000; //ms
    private final int NUM_INSTRUCTION_TYPE = 4;

    private Timer mTimer;
    private Random random = new Random();
    private int[] instructionTypes = null;
    private int steps = -1;
    private int step = -1;

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

        mPoseMachine = PoseMachine.getPoseMachine(getAssets(),
                PAFNET_MODEL_FILE,
                PAFNET_INPUT_SIZE,
                PAFNET_INPUT_NODE_NAME,
                PAFNET_OUTPUT_NODE_NAMES);
    }

    public void startGame(View view) {
        create_instructions();

        // set up the timer
        if (mTimer == null) {
            mTimer = new Timer();
            setTimerTask();
        }

        // set up the music
        if(mMediaPlayer != null) mMediaPlayer.release();
            mMediaPlayer = MediaPlayer.create(this, R.raw.carolina);
            mMediaPlayer.setLooping(true);
            // TODO: deal with the case that the song is looping
            mMediaPlayer.start();

        int duration = mMediaPlayer.getDuration();
        Log.v("MUSIC", "duration:" + duration);

        steps = duration / timeStep;
        instructionTypes = new int[steps];
        step = -1;
        for (int i = 0; i < steps; ++i){
            instructionTypes[i] = random.nextInt(NUM_INSTRUCTION_TYPE);
        }
    }

    private void setTimerTask() {
        mTimer.scheduleAtFixedRate(new MusicTask(), timeStep, timeStep);
    }

    private class MusicTask extends TimerTask {
        @Override
        public void run() {
            //update TextView
            Message message = new Message();
            message.what = 1;
            doActionHandler.sendMessage(message);
        }
    }

    private String getInstuctionStringByType(int type){
        switch (type){
            case 0:
                return getResources().getString(R.string.up_arrow);
            case 1:
                return getResources().getString(R.string.down_arrow);
            case 2:
                return getResources().getString(R.string.left_arrow);
            case 3:
                return getResources().getString(R.string.right_arrow);
            default:
                return "";
        }
    }

    private Handler doActionHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            int msgId = msg.what;
            switch (msgId) {
                case 1:
                    // do some action
                    TextView tv = (TextView) findViewById(R.id.hello);
                    ++step;
                    tv.setText(""+step);

                    for (int i = 0; i < NUM_INSTRUCTIONS; ++i){
                        if(i + step >= steps)
                            buttons_instruction_correct[i].setText("");
                        else
                            buttons_instruction_correct[i].setText(getInstuctionStringByType(instructionTypes[i+step]));
                    }

                    break;
                default:
                    break;
            }
        }
    };

    protected void create_instructions() {
        if (buttons_instruction_correct != null && buttons_instruction_infer != null) return;
        LinearLayout instructions_correct = (LinearLayout) findViewById(R.id.instructions_correct);
        buttons_instruction_correct = new Button[NUM_INSTRUCTIONS];
        for (int i = 0; i < NUM_INSTRUCTIONS; ++i) {
            buttons_instruction_correct[i] = new Button(this);
            buttons_instruction_correct[i].setTag(i);
            buttons_instruction_correct[i].setTextColor(getResources().getColor(R.color.colorAccent));
            buttons_instruction_correct[i].setText(getInstuctionStringByType(i));
            instructions_correct.addView(buttons_instruction_correct[i],
                    new LinearLayout.LayoutParams(
                            (int) getResources().getDimension(R.dimen.instruction_width),
                            (int) getResources().getDimension(R.dimen.instruction_height)));

        }

        // only need one for inferring.........
        LinearLayout instructions_infer = (LinearLayout) findViewById(R.id.instructions_infer);
        buttons_instruction_infer = new Button[1];
        buttons_instruction_infer[0] = new Button(this);
        buttons_instruction_infer[0].setTag(0);
        buttons_instruction_infer[0].setTextColor(getResources().getColor(R.color.colorAccent));
        buttons_instruction_infer[0].setText(getInstuctionStringByType(0));
        instructions_infer.addView(buttons_instruction_infer[0],
                new LinearLayout.LayoutParams(
                        (int) getResources().getDimension(R.dimen.instruction_width),
                        (int) getResources().getDimension(R.dimen.instruction_height)));

        // make the first one bigger
        buttons_instruction_correct[0].setTextSize(getResources().getDimension(R.dimen.bigTextSize));
        buttons_instruction_infer[0].setTextSize(getResources().getDimension(R.dimen.bigTextSize));
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
        if (mTimer != null) {
            mTimer.cancel();
            mTimer = null;
        }
        super.onPause();
    }

    @Override
    protected void onStop() {
        //TODO: check timer and the media_player whether works when stop and resume
        Log.d(TAG, "onStop");
        if (mTimer != null) {
            mTimer.cancel();
            mTimer = null;
        }
        super.onStop();
        if (mMediaPlayer != null) {
            mMediaPlayer.release();
            mMediaPlayer = null;
        }
    }

    @Override
    protected void onDestroy() {
        Log.d(TAG, "onDestroy");
        if (mTimer != null) {
            mTimer.cancel();
            mTimer = null;
        }
        if (mPoseMachine != null) {
            mPoseMachine.close();
        }
        super.onDestroy();
    }


    @Override
    public void onImageAvailable(final ImageReader reader) {
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
                    mOverlayView.postInvalidate();
                }
                // TODO: draft, need to debug
                int type = getInferPoseType();
                if(buttons_instruction_infer != null)
                    buttons_instruction_infer[0].setText(getInstuctionStringByType(type));
            }
        });
        Trace.endSection();
    }


    public int getInferPoseType(){
        // get pose
        float[] detectedPose = mPoseMachine.getDetectedPositions();
        // {0,  "Nose"}, {1,  "Neck"},
        // {2,  "RShoulder"}, {3,  "RElbow"}, {4,  "RWrist"},
        // {5,  "LShoulder"}, {6,  "LElbow"}, {7,  "LWrist"},
        // {8,  "RHip"}, {9,  "RKnee"}, {10, "RAnkle"},
        // {11, "LHip"}, {12, "LKnee"}, {13, "LAnkle"},
        // {14, "REye"}, {15, "LEye"}, {16, "REar"}, {17, "LEar"}

        // simple way to decide the type of pose
        float[] rWrist = {detectedPose[4*2], detectedPose[4*2+1]};
        float[] lWrist = {detectedPose[7*2], detectedPose[7*2+1]};
        float[] nose   = {detectedPose[0*2], detectedPose[0*2+1]};
        Log.v("POSE", "rWrist: " + Arrays.toString(rWrist));
        Log.v("POSE", "lWrist: " + Arrays.toString(lWrist));
        Log.v("POSE", "nose: " + Arrays.toString(nose));

        // int r = (int)detectedPose[i*2];
        // int c = (int)detectedPose[i*2+1];
        // [row, col]
        if((rWrist[1] > nose[1]) != (lWrist[1] > nose[1])){
            // wrists in the different sides
            if(rWrist[0] < nose[0] && lWrist[0] < nose[0]){
                // wrists above the nose
                return 0; // up
            } else return 1; // down
        } else if (rWrist[1] > nose[1]){ // same side
            return 3; // right
        } else { //rWrist[1] > nose[1] same side
            return 2; // left
        }

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
                float[] detectedPose = mPoseMachine.getDetectedPositions();
                lines.add(Arrays.toString(detectedPose));
                Log.v("POSE", Arrays.toString(detectedPose));


                ImageView mImageView = (ImageView) findViewById(R.id.crop_image);

                // draw blue pixels at the pose joints
                Bitmap poseBitmap = Bitmap.createScaledBitmap(mCroppedBitmap, PAFNET_OUTPUT_SIZE, PAFNET_OUTPUT_SIZE, false);
                int[] pixels = new int[poseBitmap.getHeight()*poseBitmap.getWidth()];
                poseBitmap.getPixels(pixels, 0, poseBitmap.getWidth(), 0, 0, poseBitmap.getWidth(), poseBitmap.getHeight());
                for (int i = 0; i < detectedPose.length/2; ++i){
                    int r = (int)detectedPose[i*2];
                    int c = (int)detectedPose[i*2+1];
                    pixels[r*PAFNET_OUTPUT_SIZE+c] = Color.BLUE;
                }
                poseBitmap.setPixels(pixels, 0, poseBitmap.getWidth(), 0, 0, poseBitmap.getWidth(), poseBitmap.getHeight());
                poseBitmap = Bitmap.createScaledBitmap(poseBitmap, PAFNET_INPUT_SIZE, PAFNET_INPUT_SIZE, false);

                mImageView.setImageBitmap(poseBitmap);

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
        mCroppedBitmap = Bitmap.createBitmap(PAFNET_INPUT_SIZE, PAFNET_INPUT_SIZE,
                Bitmap.Config.ARGB_8888);

        mFrameToCropTransform = ImageUtils.getTransformationMatrix(mPreviewWidth, mPreviewHeight,
                PAFNET_INPUT_SIZE, PAFNET_INPUT_SIZE, mSensorOrientation, false);
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
