package edu.temple.mar_security.ml_kit.tflite;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

import edu.temple.mar_security.ml_kit.face_detection.FaceProcessor;

public class AgeClassifier  extends Classifier {
    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 0f;

    private static final float IMAGE_STD = 255f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code ClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public AgeClassifier (Activity activity )
            throws IOException {
        super(activity );
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.

        return "age224.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "age_label.txt";
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

}

