package unc.cs.v3d.dancedanceconvolution;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Created by kewang on 4/20/17.
 */

public class PoseMachine {
    private static final String TAG = PoseMachine.class.getSimpleName();
    private TensorFlowInferenceInterface mTFinferenceInterface;
    private boolean mIfLogStats = false;

    // Network config
    private String mInputName;
    private int mInputWidth;
    private int mInputHeight;

    // pre-allocated buffers
    private int[] mInputIntBuffer;
    private float[] mInputFloatBuffer;
    private float[] mOutputHeatmap;
    private String[] mOutputNames;
    private float[] mDetectedPositions;
    private float[] mDetectionScores;

    int getTensorSize(final Operation op) {
        final int outputTensorRank = op.output(0).shape().numDimensions();
        int outputSize = 1;
        String outputDim = "[";
        for (int i = 0; i < outputTensorRank; ++i) {
            outputSize *= op.output(0).shape().size(i);
            outputDim += op.output(0).shape().size(i);
            if (i < outputTensorRank - 1) {
                outputDim += "*";
            }
        }
        outputDim += "]";
        Log.i(TAG, "Output layer " + op.name() + " size:" + outputDim);

        return outputSize;
    }

    // Singleton pattern
    public static PoseMachine getPoseMachine(AssetManager assetManager, String modelFilename,
                                             int inputSize, String inputName, String[] outputNames) {
        PoseMachine pm = new PoseMachine();
        pm.mInputName = inputName;
        pm.mOutputNames = outputNames;

        try {
            pm.mTFinferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        } catch (Exception e) {
            Log.e(TAG, "Exception:", e);
            throw e;
        }
        Log.e(TAG, "" + outputNames.length);
        final Operation heatmapOp = pm.mTFinferenceInterface.graphOperation(pm.mOutputNames[0]);
        final int heatmapOutputTensorSize = pm.getTensorSize(heatmapOp);
        pm.mOutputHeatmap = new float[heatmapOutputTensorSize];
        Log.i(TAG, "Allocated output buffer size for heatmap:" + heatmapOutputTensorSize);

        pm.mInputWidth = inputSize;
        pm.mInputHeight = inputSize;
        pm.mInputIntBuffer = new int[pm.mInputWidth * pm.mInputHeight];
        pm.mInputFloatBuffer = new float[pm.mInputWidth * pm.mInputHeight * 3];
        // fixme(fswangke): refactor hard-coded 18
        pm.mDetectionScores = new float[18];
        pm.mDetectedPositions = new float[18*2];

        return pm;
    }

    public float[] getHeatmap() {
        return mOutputHeatmap;
    }

    public float[] getDetectedPositions() {
        return mDetectedPositions;
    }

    public void inferPose(Bitmap bitmap) {
        Trace.beginSection("inferPose");

        Trace.beginSection("preprocessBitmap");
        // TODO: faster way of preprocessing images?
        bitmap.getPixels(mInputIntBuffer, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < mInputIntBuffer.length; ++i) {
            final int val = mInputIntBuffer[i];
            mInputFloatBuffer[i * 3 + 0] = (float) ((val >> 16) & 0xFF) / 255.0f - 0.5f;
            mInputFloatBuffer[i * 3 + 1] = (float) ((val >> 8) & 0xFF) / 255.0f - 0.5f;
            mInputFloatBuffer[i * 3 + 2] = (float) (val & 0xFF) / 255.0f - 0.5f;
        }
        Trace.endSection(); // preprocessBitmap

        mTFinferenceInterface.feed(mInputName, mInputFloatBuffer, 1, mInputHeight, mInputWidth, 3);
        mTFinferenceInterface.run(mOutputNames, mIfLogStats);
        mTFinferenceInterface.fetch(mOutputNames[0], mOutputHeatmap);

        // TODO: simple arg-max hack to pick one person
        // TensorFlow output tensor shape: 1x46x46x19
        for(int partId = 0; partId < 18; ++partId) {
            mDetectionScores[partId] = Float.MIN_VALUE;
        }
        // fixme: refactor hard-coded 28
        // output heatmap tensor size is 1x28x28x19 for 224x224x3 input images
        for(int row = 0; row < 28; ++row) {
            for(int col = 0; col < 28; ++col) {
                for(int part = 0; part < 18; ++part) {
                    int index = part + (col + row * 28) * 19;
                    if(mOutputHeatmap[index] > mDetectionScores[part]) {
                        mDetectionScores[part] = mOutputHeatmap[index];
                        mDetectedPositions[part * 2 + 0] = row;
                        mDetectedPositions[part * 2 + 1] = col;
                    }
                }
            }
        }

        Trace.endSection(); // inferPose
    }

    void enableStatLogging(final boolean debug) {
        mIfLogStats = debug;
    }

    String getStatString() {
        if (mTFinferenceInterface != null) {
            return mTFinferenceInterface.getStatString();
        } else {
            return "";
        }
    }

    void close() {
        mTFinferenceInterface.close();
    }
}
