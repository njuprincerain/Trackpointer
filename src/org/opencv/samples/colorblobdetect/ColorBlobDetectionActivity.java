package org.opencv.samples.colorblobdetect;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.text.Selection;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.Window;
import android.view.WindowManager;
import android.widget.EditText;

public class ColorBlobDetectionActivity extends Activity implements
		OnTouchListener, CvCameraViewListener2 {
	private static final String TAG = "OCVSample::Activity";

	private boolean mIsColorSelected = false;
	private Mat mRgba;
	private Scalar mBlobColorRgba;
	private Scalar mBlobColorHsv;
	private ColorBlobDetector mDetector;
	private Mat mSpectrum;
	private Size SPECTRUM_SIZE;
	private Scalar CONTOUR_COLOR;

	private CameraBridgeViewBase mOpenCvCameraView;
	private EditText editText;

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
				mOpenCvCameraView
						.setOnTouchListener(ColorBlobDetectionActivity.this);
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public ColorBlobDetectionActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.color_blob_detection_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
		editText = (EditText) findViewById(R.id.sample_edit_text);
		// Selection.moveLeft(editText.getText(), editText.getLayout());
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		mDetector = new ColorBlobDetector();
		mSpectrum = new Mat();
		mBlobColorRgba = new Scalar(255);
		mBlobColorHsv = new Scalar(255);
		SPECTRUM_SIZE = new Size(200, 64);
		CONTOUR_COLOR = new Scalar(255, 0, 0, 255);
	}

	public void onCameraViewStopped() {
		mRgba.release();
	}

	public boolean onTouch(View v, MotionEvent event) {
		int cols = mRgba.cols();
		int rows = mRgba.rows();

		int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
		int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

		int x = (int) event.getX() - xOffset;
		int y = (int) event.getY() - yOffset;

		Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

		if ((x < 0) || (y < 0) || (x > cols) || (y > rows))
			return false;

		Rect touchedRect = new Rect();

		touchedRect.x = (x > 4) ? x - 4 : 0;
		touchedRect.y = (y > 4) ? y - 4 : 0;

		touchedRect.width = (x + 4 < cols) ? x + 4 - touchedRect.x : cols
				- touchedRect.x;
		touchedRect.height = (y + 4 < rows) ? y + 4 - touchedRect.y : rows
				- touchedRect.y;

		Mat touchedRegionRgba = mRgba.submat(touchedRect);

		Mat touchedRegionHsv = new Mat();
		Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv,
				Imgproc.COLOR_RGB2HSV_FULL);

		// Calculate average color of touched region
		mBlobColorHsv = Core.sumElems(touchedRegionHsv);
		int pointCount = touchedRect.width * touchedRect.height;
		for (int i = 0; i < mBlobColorHsv.val.length; i++)
			mBlobColorHsv.val[i] /= pointCount;

		mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

		Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", "
				+ mBlobColorRgba.val[1] + ", " + mBlobColorRgba.val[2] + ", "
				+ mBlobColorRgba.val[3] + ")");

		mDetector.setHsvColor(mBlobColorHsv);

		Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

		mIsColorSelected = true;

		touchedRegionRgba.release();
		touchedRegionHsv.release();

		return false; // don't need subsequent touch events
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		mRgba = inputFrame.rgba();

		if (mIsColorSelected) {
			mDetector.process(mRgba);
			List<MatOfPoint> contours = mDetector.getContours();
			Log.e(TAG, "Contours count: " + contours.size());
			Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR);

			Mat colorLabel = mRgba.submat(4, 68, 4, 68);
			colorLabel.setTo(mBlobColorRgba);

			Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70,
					70 + mSpectrum.cols());
			mSpectrum.copyTo(spectrumLabel);
		}
		Mat smMat = getSmallerMat(mRgba);
		Imgproc.cvtColor(smMat, smMat, Imgproc.COLOR_BGR2GRAY);
		int row = smMat.width();
		int col = smMat.height();
		int rDiv1 = row / 3;
		int rDiv2 = row * 2 / 3;
		int colDiv1 = col / 3;
		int colDiv2 = col * 2 / 3;
		double totalBrgt = 0;
		double upBrgt = 0;
		double leftBrgt = 0;
		double rightBrgt = 0;
		double downBrgt = 0;
		ArrayList<Double> brgtList = new ArrayList<Double>();
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double[] dimensions = smMat.get(i, j);
				if (dimensions != null) {
					double brgt = dimensions[0];
					totalBrgt += brgt;
					brgtList.add(brgt);
					// up
					if (i <= rDiv1 && j >= colDiv1 && j <= colDiv2) {
						upBrgt += brgt;
					}
					// down
					if (i >= rDiv2 && j >= colDiv1 && j <= colDiv2) {
						downBrgt += brgt;
					}
					// left
					if (j <= colDiv1 && i >= rDiv1 && i <= rDiv2) {
						leftBrgt += brgt;
					}
					// right
					if (j >= colDiv2 && i >= rDiv1 && i <= rDiv2) {
						rightBrgt += brgt;
					}
				}
			}
		}
		double verticalDiff = 100 * (upBrgt - downBrgt) / totalBrgt;
		double horizontalDiff = 100 * (leftBrgt - rightBrgt) / totalBrgt;
		// Log.e(TAG, "Total" + totalBrgt);
		double std = std(brgtList);
		// Log.e(TAG, "std " + std);
		if (std < 5000){
		    Log.e(TAG, "Diff (" + horizontalDiff + ", " + verticalDiff + ")");
		    setCursor(horizontalDiff, verticalDiff);
		}
		return mRgba;
	}
	
	private void setCursor(final double horiDiff, final double vertiDiff){
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				if (horiDiff > 14){
					Selection.moveRight(editText.getText(), editText.getLayout());
				}
				if (horiDiff < -13){
					Selection.moveLeft(editText.getText(), editText.getLayout());
				}
				if (vertiDiff > 18){
					Selection.moveDown(editText.getText(), editText.getLayout());
				}
				if (vertiDiff < 0){
					Selection.moveUp(editText.getText(), editText.getLayout());
				}
			}
		});
	}

	private Mat getSmallerMat(Mat m) {
		int row = m.width();
		int col = m.height();
		Size size = new Size(row / 5, col / 5);
		Mat newMat = new Mat();
		Imgproc.resize(m, newMat, size);
		return newMat;
	}

	private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
		Mat pointMatRgba = new Mat();
		Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
		Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL,
				4);

		return new Scalar(pointMatRgba.get(0, 0));
	}
	
	private double std(ArrayList<Double> list){
		double average = avg(list);
		int sum = 0;
		for (int i = 0; i < list.size(); i++){
			double curDiff = list.get(i) - average;
			sum += curDiff * curDiff;
		}
		return Math.sqrt(sum);
	}
	
	private double avg(ArrayList<Double> list){
		double sum = 0;
		for (int i = 0; i < list.size(); i++){
			sum += list.get(i);
		}
		return sum / list.size();
	}
}
