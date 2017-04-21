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
    private int imageMean;
    private int imageStd;

    // pre-allocated buffers
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    // Singleton pattern

    public static PoseMachine getPoseMachine(AssetManager assetManager, String modelFilename, int inputSize, int imageMean, int imageStd, String inputName, String outputName) {
        PoseMachine pm = new PoseMachine();
        pm.inputName = inputName;
        pm.outputName = outputName;

        try {
            pm.mTFinferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        } catch (Exception e) {
            Log.e(TAG, "Exception:", e);
            throw e;
        }
        final Operation operation = pm.mTFinferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Output layer size:" + numClasses);

        pm.inputSize = inputSize;
        pm.imageMean = imageMean;
        pm.imageStd = imageStd;

        pm.outputNames = new String[]{outputName};
        pm.intValues = new int[inputSize * inputSize];
        pm.floatValues = new float[inputSize * inputSize * 3];
        pm.outputs = new float[numClasses];

        return pm;
    }

    public void inferPose(Bitmap bitmap) {
        Trace.beginSection("inferPose");

        Trace.beginSection("preprocessBitmap");
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        Trace.endSection(); // preprocessBitmap

        mTFinferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        mTFinferenceInterface.run(outputNames, mIfLogStats);
        mTFinferenceInterface.fetch(outputName, outputs);

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
