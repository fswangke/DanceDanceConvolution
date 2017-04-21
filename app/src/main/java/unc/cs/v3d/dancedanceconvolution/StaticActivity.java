package unc.cs.v3d.dancedanceconvolution;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.os.UserHandle;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Vector;

public class StaticActivity extends AppCompatActivity {
    private static final String TAG = StaticActivity.class.getSimpleName();
    private ImageView mImageView;
    private Bitmap mBitmap;
    private final int PAF_NET_SIZE = 224;
    private PoseMachine mPoseMachine;
    private TextView mTextView;
    private static final String PAFNET_MODEL_FILE = "file:///android_asset/paf_net_eightbit.pb";
    private static final String mImageFilePath = "ski.jpg";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_static);

        mImageView = (ImageView) findViewById(R.id.static_image_view);
        AssetManager assetManager = getAssets();

        InputStream istr = null;
        try {
            istr = assetManager.open(mImageFilePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap rawImage = BitmapFactory.decodeStream(istr);
        mBitmap = Bitmap.createScaledBitmap(rawImage, PAF_NET_SIZE, PAF_NET_SIZE, false);

        mImageView.setImageBitmap(mBitmap);
        startTFInferenceThread();

        // init pose machine
        String[] outputNames = new String[]{"conv5_5_CPM_L1", "conv5_5_CPM_L2"};
        mPoseMachine = PoseMachine.getPoseMachine(getAssets(), PAFNET_MODEL_FILE, PAF_NET_SIZE, "image", outputNames);
        mPoseMachine.enableStatLogging(true);
        mTextView = (TextView) findViewById(R.id.tv_tf_stats);
        mTextView.setTextSize(8.0f);
    }

    @Override
    protected void onPause() {
        stopTFInferenceThread();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        startTFInferenceThread();
        runInferenceInBackground(new Runnable() {
            @Override
            public void run() {
                final long startTime = SystemClock.uptimeMillis();
                final int COUNT = 5;
                float avgTime = 0;
                // infer
                Log.i(TAG, "Started inference.");
                for (int i = 0; i < COUNT; ++i) {
                    mPoseMachine.inferPose(mBitmap);
                    long totalTime = SystemClock.uptimeMillis() - startTime;
                    avgTime = (totalTime) / (float) (i + 1);
                    Log.i(TAG, "Iteration " + (i + 1) + " avgTime: " + avgTime + "ms");
                }

            }
        });
    }

    private HandlerThread mHandlerThread;
    private Handler mHandler;

    private void startTFInferenceThread() {
        mHandlerThread = new HandlerThread("TFinferencethread");
        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper());
    }

    private void stopTFInferenceThread() {
        mHandlerThread.quitSafely();

        try {
            mHandlerThread.join();
            mHandlerThread = null;
            mHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void runInferenceInBackground(final Runnable r) {
        if (mHandler != null) {
            mHandler.post(r);
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            Log.v(TAG, "Trying to trigger overlay display");
            mTextView.setText(mPoseMachine.getStatString());
            mImageView.setImageBitmap(mBitmap);
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }
}
