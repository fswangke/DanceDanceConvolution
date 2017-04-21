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
    private String inputName;
    private String outputName;
    private int inputSize;
    private int inputWidth;
    private int inputHeight;
    private int imageMean;
    private int imageStd;

    // pre-allocated buffers
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    // Singleton pattern
    public static PoseMachine getPoseMachine(AssetManager assetManager, String modelFilename,
                                             int inputSize, String inputName, String[] outputNames) {
        PoseMachine pm = new PoseMachine();
        pm.inputName = inputName;
        pm.outputNames = outputNames;

        try {
            pm.mTFinferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        } catch (Exception e) {
            Log.e(TAG, "Exception:", e);
            throw e;
        }
        int maxOutputSize = Integer.MIN_VALUE;
        for (final String outputName : pm.outputNames) {
            final Operation operation = pm.mTFinferenceInterface.graphOperation(outputName);
            final int outputSize = (int) operation.output(0).shape().size(1);
            Log.i(TAG, "Output layer " + outputName + " size:" + outputSize);
            maxOutputSize = Math.max(maxOutputSize, outputSize);
        }
        Log.i(TAG, "Allocated buffer size for output:" + maxOutputSize);

        pm.inputWidth = inputSize;
        pm.inputHeight = inputSize;
        pm.intValues = new int[pm.inputWidth * pm.inputHeight];
        pm.floatValues = new float[pm.inputWidth * pm.inputHeight * 3];
        pm.outputs = new float[maxOutputSize];

        return pm;
    }

    public void inferPose(Bitmap bitmap) {
        Trace.beginSection("inferPose");

        Trace.beginSection("preprocessBitmap");
        // TODO: faster way of preprocessing images?
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (float) ((val >> 16) & 0xFF) / 255.0f - 0.5f;
            floatValues[i * 3 + 1] = (float) ((val >> 8) & 0xFF) / 255.0f - 0.5f;
            floatValues[i * 3 + 2] = (float) (val & 0xFF) / 255.0f - 0.5f;
        }
        Trace.endSection(); // preprocessBitmap

        mTFinferenceInterface.feed(inputName, floatValues, 1, inputHeight, inputWidth, 3);
        mTFinferenceInterface.run(outputNames, mIfLogStats);
        // TODO: fetch the result and perform parsing to get skelton
//        mTFinferenceInterface.fetch(outputName, outputs);

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
