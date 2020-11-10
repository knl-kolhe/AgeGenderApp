package edu.temple.mar_security.ml_kit.face_detection;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Rect;
import android.media.Image;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;

import edu.temple.mar_security.ml_kit.MainActivity;
import edu.temple.mar_security.ml_kit.tflite.Classifier;
import edu.temple.mar_security.ml_kit.utils.FileIOUtil;

import edu.temple.mar_security.ml_kit.tflite.AgeClassifier;
import edu.temple.mar_security.ml_kit.Logger;

public class FaceProcessor implements FaceAnalyzer.FaceAnalysisListener {

    private static final String TAG = MainActivity.TAG;

    private Context context;
    private Activity mActivity;

    public FaceProcessor(Activity parent) {
        this.context = parent.getApplicationContext();
        this.mActivity = parent;
    }

    Interpreter age_tflite, gender_tflite;

//    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
//        AssetFileDescriptor fileDescriptor = assets.openFd(MODEL_FILE);
//        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
//        FileChannel fileChannel = inputStream.getChannel();
//        long startOffset = fileDescriptor.getStartOffset();
//        long declaredLength = fileDescriptor.getDeclaredLength();
//        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
//    }


    @Override
    public void facesFound(Image image, int rotation, List<Rect> boundingBoxes) {
        Log.i(TAG, "Received image with " + boundingBoxes.size()
                + " bounding boxes and rotation: " + rotation);

        try {
            for (Rect boundingBox : boundingBoxes) {
                // extract the faces according to the bounding box coordinates
                Bitmap faceImg = cropFace(image, rotation, boundingBox);

                // TODO - scale the face snippets to a useful size
                faceImg = scaleFace(faceImg);

                // TODO - extract RGB matrices
                ByteBuffer model_input = extractFeatureVectors(faceImg);

                // TODO - feed inputs to Kunal's CV modules
                performAdditionalProcessing(faceImg);
            }
        } catch (Exception ex) {
            Log.e(TAG, "Something went wrong while attempting to "
                    + "process detected faces!", ex);
        }
    }

    // ----------------------------------------------------------------------------------
    //      PRIVATE METHODS
    // ----------------------------------------------------------------------------------

    private Bitmap cropFace(Image image, int rotation, Rect boundingBox) {
        Bitmap bitmap = FileIOUtil.toBitmap(image, rotation);
        Bitmap croppedBitmap = FileIOUtil.crop(bitmap, boundingBox);
        FileIOUtil.writeToFile(context, croppedBitmap);
        return croppedBitmap;
    }

    private Bitmap scaleFace(Bitmap faceImg) {

        return Bitmap.createScaledBitmap(
                faceImg,
                224,
                224,
                false
        );

    }

    private ByteBuffer extractFeatureVectors(Bitmap faceImg) {
        Bitmap bitmap = Bitmap.createScaledBitmap(faceImg, 224, 224, true);
        ByteBuffer input = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int px = bitmap.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = (r - 127) / 255.0f;
                float gf = (g - 127) / 255.0f;
                float bf = (b - 127) / 255.0f;

                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }
        return input;
    }

    private void performAdditionalProcessing(Bitmap model_input) {
        // empty
        Logger LOGGER = new Logger();
        Classifier classifier = null;
        try {
            classifier = new AgeClassifier(mActivity);
        } catch (IOException e) {
            LOGGER.e("Loading Age classifier");
        }
        final List<Classifier.Recognition> results =
                classifier.recognizeImage(model_input);
        Log.i("Hello wtf", Arrays.toString(results.toArray()));


//        Log.d("Path", context.getApplicationInfo().dataDir);
//        try {
//            String age_path = "age224.tflite";
//            File age_model = new File(String.valueOf(context.getAssets().open( age_path)));
//
//            age_tflite = new Interpreter(age_model);
//
//            String gender_path = "gender_MobileNetV2.tflite";
//            File gender_model = new File(String.valueOf(context.getAssets().open( gender_path)));
//
//            gender_tflite = new Interpreter(gender_model);
//
//
//
//        }
//        catch(Exception e){
//            e.printStackTrace();
//        }
//
//        int bufferSize = 1 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
//        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
//        age_tflite.run(model_input, modelOutput);
//        Log.d("Model inference result", modelOutput.toString());
//
//        bufferSize = 1 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
//        modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
//        gender_tflite.run(model_input, modelOutput);
//        Log.d("Model inference result", modelOutput.toString());
    }

}
