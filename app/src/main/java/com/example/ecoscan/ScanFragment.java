package com.example.ecoscan;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder; // Importante
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ScanFragment extends Fragment {

    private static final String TAG = "EcoScanApp";
    private static final String MODEL_FILE = "best.tflite";
    private static final String LABEL_FILE = "labels.txt";
    private static final float CONFIDENCE_THRESHOLD = 0.2f;
    private static final int INPUT_SIZE = 640;

    private ImageView imageView;
    private Button buttonCamera;
    private Button buttonGallery;
    private Button buttonAnalyze;
    private TextView textViewResult;

    private Interpreter interpreter;
    private List<String> labels = new ArrayList<>();
    private int inputWidth;
    private int inputHeight;
    private int outputNumClasses;
    private int outputNumProposals;

    private ImageProcessor imageProcessor;

    // Launchers
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<Intent> galleryLauncher;
    private ActivityResultLauncher<String> permissionLauncher;

    private Bitmap bitmapToAnalyze;
    private Uri cameraImageUri;

    // Classe interna (exatamente como era)
    private static class DisposalDetails {
        String objectName;
        String binName;
        String binDescription;
        int binColorRes;

        DisposalDetails(String objectName, String binName, String binDescription, int binColorRes) {
            this.objectName = objectName;
            this.binName = binName;
            this.binDescription = binDescription;
            this.binColorRes = binColorRes;
        }
    }

    // --- Métodos de Ciclo de Vida do Fragment ---

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setupLaunchers();
        try {
            loadLabels();
            loadModel();
            setupImageProcessor();
            Log.d(TAG, "TensorFlow Lite inicializado com sucesso.");
        } catch (IOException e) {
            Log.e(TAG, "Falha ao inicializar o TensorFlow Lite.", e);
            Toast.makeText(requireContext(), "Não foi possível carregar o modelo ou labels.", Toast.LENGTH_LONG).show();
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_scan, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        imageView = view.findViewById(R.id.imageView);
        buttonCamera = view.findViewById(R.id.buttonCamera);
        buttonGallery = view.findViewById(R.id.buttonGallery);
        buttonAnalyze = view.findViewById(R.id.buttonAnalyze);
        textViewResult = view.findViewById(R.id.textViewResult);
        textViewResult.setText("");
        buttonCamera.setOnClickListener(v -> checkCameraPermissionAndOpenCamera());
        buttonGallery.setOnClickListener(v -> openGallery());
        buttonAnalyze.setOnClickListener(v -> analyzeImage());
        buttonAnalyze.setEnabled(false);
    }

    // --- Restante do seu código (copiado 1:1, com ajustes de CONTEXTO) ---

    private void analyzeImage() {
        if (bitmapToAnalyze != null) {
            textViewResult.setText("Analisando...");
            Log.d(TAG, "Iniciando detecção...");
            imageView.setImageBitmap(bitmapToAnalyze);
            detectObjects(bitmapToAnalyze);
        } else {
            Toast.makeText(requireContext(), "Selecione uma imagem da câmera ou galeria primeiro.", Toast.LENGTH_SHORT).show();
            textViewResult.setText("Nenhuma imagem selecionada para análise.");
        }
    }

    private void loadLabels() throws IOException {
        AssetManager assetManager = requireContext().getAssets();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(LABEL_FILE)));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
    }

    private void loadModel() throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(requireContext(), MODEL_FILE);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(tfliteModel, options);
        int[] inputShape = interpreter.getInputTensor(0).shape();
        inputWidth = inputShape[1];
        inputHeight = inputShape[2];
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputNumClasses = outputShape[1] - 4;
        outputNumProposals = outputShape[2];
    }

    private void setupImageProcessor() {
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0f, 255f))
                .build();
    }

    // --- MODIFICADO: Carregamento de Imagem em Alta Resolução ---

    /**
     * Carrega um Bitmap de uma Uri.
     * *** CORRIGIDO PARA EVITAR CRASH DE HARDWARE BITMAP ***
     */
    private Bitmap loadBitmapFromUri(Uri uri) throws IOException {
        ContentResolver resolver = requireContext().getContentResolver();
        ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);

        // Esta é a correção:
        return ImageDecoder.decodeBitmap(source, (decoder, info, s) -> {
            // Força o bitmap a ser alocado na memória de "software" (RAM)
            // Isso impede o crash do TFLite e do Canvas.
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);

            // Pede que o bitmap seja mutável (bom para desenhar)
            decoder.setMutableRequired(true);
        });
    }

    /**
     * Cria uma URI segura para a câmera salvar a foto em alta resolução.
     */
    private Uri createImageUri() {
        File imagePath = new File(requireContext().getCacheDir(), "images");
        if (!imagePath.exists()) imagePath.mkdirs();

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        File newFile = new File(imagePath, "IMG_" + timeStamp + ".jpg");

        return FileProvider.getUriForFile(requireContext(), requireContext().getPackageName() + ".provider", newFile);
    }

    private void setupLaunchers() {
        permissionLauncher = registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
            if (isGranted) openCamera();
            else Toast.makeText(requireContext(), "Permissão da câmera negada.", Toast.LENGTH_SHORT).show();
        });

        cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == android.app.Activity.RESULT_OK) {
                if (cameraImageUri != null) {
                    try {
                        bitmapToAnalyze = loadBitmapFromUri(cameraImageUri);
                        imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                        imageView.setImageBitmap(bitmapToAnalyze);
                        textViewResult.setText("Imagem carregada. Clique em 'Analisar'.");
                        buttonAnalyze.setEnabled(true);
                    } catch (IOException e) {
                        Log.e(TAG, "Erro ao carregar imagem da câmera (Uri).", e);
                        textViewResult.setText("Erro ao carregar foto.");
                    }
                }
            }
        });

        galleryLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == android.app.Activity.RESULT_OK && result.getData() != null) {
                Uri imageUri = result.getData().getData();
                if (imageUri != null) {
                    try {
                        bitmapToAnalyze = loadBitmapFromUri(imageUri);
                        imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                        imageView.setImageBitmap(bitmapToAnalyze);
                        textViewResult.setText("Imagem carregada. Clique em 'Analisar'.");
                        buttonAnalyze.setEnabled(true);
                    } catch (IOException e) {
                        Log.e(TAG, "Erro ao carregar imagem da galeria.", e);
                        textViewResult.setText("Erro ao carregar imagem da galeria.");
                    }
                }
            }
        });
    }

    private void checkCameraPermissionAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void openCamera() {
        cameraImageUri = createImageUri();
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri);
        cameraLauncher.launch(takePictureIntent);
    }

    private void openGallery() {
        Intent pickIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryLauncher.launch(pickIntent);
    }

    // --- Métodos de Detecção (iguais) ---

    private void detectObjects(Bitmap bitmap) {
        if (interpreter == null || labels.isEmpty()) {
            Toast.makeText(requireContext(), "Detector não foi inicializado.", Toast.LENGTH_SHORT).show();
            return;
        }

        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
        tensorImage = imageProcessor.process(tensorImage);
        ByteBuffer inputBuffer = tensorImage.getBuffer();

        float[][][] outputArray = new float[1][4 + outputNumClasses][outputNumProposals];

        try {
            interpreter.run(inputBuffer, outputArray);
        } catch (Exception e) {
            Log.e(TAG, "Erro na execução do modelo TFLite.", e);
            Toast.makeText(requireContext(), "Erro na execução do modelo.", Toast.LENGTH_LONG).show();
            return;
        }

        List<Detection> detections = postProcessYolo(outputArray[0], bitmap.getWidth(), bitmap.getHeight());
        displayDetectionResult(detections, bitmap);
    }

    // Classe interna (igual)
    private static class Detection {
        RectF boundingBox;
        String label;
        float confidence;
        Detection(RectF boundingBox, String label, float confidence) {
            this.boundingBox = boundingBox;
            this.label = label;
            this.confidence = confidence;
        }
    }

    // postProcessYolo, nonMaxSuppression, calculateIoU (iguais)
    private List<Detection> postProcessYolo(float[][] output, int originalWidth, int originalHeight) {
        float[][] transposedOutput = new float[outputNumProposals][4 + outputNumClasses];
        for (int i = 0; i < 4 + outputNumClasses; i++) {
            for (int j = 0; j < outputNumProposals; j++) {
                transposedOutput[j][i] = output[i][j];
            }
        }
        List<Detection> allDetections = new ArrayList<>();
        for (int i = 0; i < outputNumProposals; i++) {
            float[] proposal = transposedOutput[i];
            int bestClassIndex = -1;
            float maxScore = 0.0f;
            for (int j = 4; j < 4 + outputNumClasses; j++) {
                if (proposal[j] > maxScore) {
                    maxScore = proposal[j];
                    bestClassIndex = j - 4;
                }
            }
            if (maxScore > CONFIDENCE_THRESHOLD) {
                float cx = proposal[0], cy = proposal[1], w = proposal[2], h = proposal[3];
                float left = cx - (w / 2f), top = cy - (h / 2f), right = cx + (w / 2f), bottom = cy + (h / 2f);
                String label = (bestClassIndex >= 0 && bestClassIndex < labels.size()) ? labels.get(bestClassIndex) : "Desconhecido";
                RectF boundingBox = new RectF(left / INPUT_SIZE, top / INPUT_SIZE, right / INPUT_SIZE, bottom / INPUT_SIZE);
                allDetections.add(new Detection(boundingBox, label, maxScore));
            }
        }
        return nonMaxSuppression(allDetections);
    }

    private List<Detection> nonMaxSuppression(List<Detection> allDetections) {
        List<Detection> nmsList = new ArrayList<>();
        float IOU_THRESHOLD = 0.45f;
        allDetections.sort(Comparator.comparingDouble((Detection d) -> d.confidence).reversed());
        for (Detection detection : allDetections) {
            boolean keep = true;
            for (Detection nmsDetection : nmsList) {
                float iou = calculateIoU(detection.boundingBox, nmsDetection.boundingBox);
                if (iou > IOU_THRESHOLD) {
                    keep = false;
                    break;
                }
            }
            if (keep) nmsList.add(detection);
        }
        return nmsList;
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
        float xA = Math.max(boxA.left, boxB.left), yA = Math.max(boxA.top, boxB.top);
        float xB = Math.min(boxA.right, boxB.right), yB = Math.min(boxA.bottom, boxB.bottom);
        float interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        float boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);
        float unionArea = (boxAArea + boxBArea - interArea);
        return interArea / unionArea;
    }

    // --- Métodos de Resultado (com ajuste de CONTEXTO) ---

    private void displayDetectionResult(List<Detection> detections, Bitmap originalBitmap) {
        if (detections.isEmpty()) {
            textViewResult.setText("Nenhum objeto reconhecido. Tente novamente.");
            imageView.setImageBitmap(originalBitmap);
            return;
        }
        Detection bestDetection = detections.get(0);
        DisposalDetails details = getDisposalDetails(bestDetection.label);
        textViewResult.setText("Resultado encontrado para: " + details.objectName);
        imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        drawDetectionBox(originalBitmap, bestDetection);

        showResultDialog(details);
    }

    private void showResultDialog(DisposalDetails details) {
        LayoutInflater inflater = LayoutInflater.from(requireContext());
        View dialogView = inflater.inflate(R.layout.dialog_result_layout, null);
        TextView textMaterialName = dialogView.findViewById(R.id.textMaterialName);
        TextView textBinName = dialogView.findViewById(R.id.textBinName);
        TextView textBinType = dialogView.findViewById(R.id.textBinType);
        View viewBinColor = dialogView.findViewById(R.id.viewBinColor);
        textMaterialName.setText(details.objectName);
        textBinName.setText(details.binName);
        textBinType.setText(details.binDescription);
        viewBinColor.setBackgroundColor(ContextCompat.getColor(requireContext(), details.binColorRes));
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setView(dialogView);
        builder.setPositiveButton("Fechar", (dialog, which) -> {
            dialog.dismiss();
            textViewResult.setText("Pronto para nova análise.");
        });
        AlertDialog dialog = builder.create();
        if (dialog.getWindow() != null) {
            dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }
        dialog.show();
    }

    private DisposalDetails getDisposalDetails(String label) {
        switch (label.toLowerCase(Locale.ROOT)) {
            case "plastic": return new DisposalDetails("Plástico", "Lixeira Vermelha", "Lixo Reciclável", R.color.lixeira_vermelho);
            case "paper": return new DisposalDetails("Papel", "Lixeira Azul", "Lixo Reciclável", R.color.lixeira_azul);
            case "cardboard": return new DisposalDetails("Papelão", "Lixeira Azul", "Lixo Reciclável", R.color.lixeira_azul);
            case "metal": return new DisposalDetails("Metal", "Lixeira Amarela", "Lixo Reciclável", R.color.lixeira_amarelo);
            case "glass": return new DisposalDetails("Vidro", "Lixeira Verde", "Lixo Reciclável", R.color.lixeira_verde);
            case "biodegradable": return new DisposalDetails("Orgânico", "Lixeira Marrom", "Lixo Orgânico / Compostagem", R.color.lixeira_marrom);
            default: return new DisposalDetails("Desconhecido", "Lixeira Cinza", "Lixo Comum / Não Reciclável", R.color.lixeira_cinza);
        }
    }

    private void drawDetectionBox(Bitmap originalBitmap, Detection bestDetection) {
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.parseColor("#4CAF50"));
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(Math.max(mutableBitmap.getWidth(), mutableBitmap.getHeight()) / 100f);
        RectF scaledBox = new RectF(
                bestDetection.boundingBox.left * originalBitmap.getWidth(),
                bestDetection.boundingBox.top * originalBitmap.getHeight(),
                bestDetection.boundingBox.right * originalBitmap.getWidth(),
                bestDetection.boundingBox.bottom * originalBitmap.getHeight()
        );
        canvas.drawRect(scaledBox, paint);
        imageView.setImageBitmap(mutableBitmap);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close();
        }
    }
}