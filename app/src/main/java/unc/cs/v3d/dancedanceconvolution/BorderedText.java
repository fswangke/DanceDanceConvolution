package unc.cs.v3d.dancedanceconvolution;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;

import java.util.Vector;

/**
 * Created by kewang on 4/20/17.
 */

public class BorderedText {
    private final Paint mInteriorPaint;
    private final Paint mExteriorPaint;
    private final float mTextSize;

    public BorderedText(final float textSize) {
        this(Color.WHITE, Color.BLACK, textSize);
    }

    public BorderedText(final int interiorColor, final int exteriorColor, final float textSize) {
        mInteriorPaint = new Paint();
        mInteriorPaint.setTextSize(textSize);
        mInteriorPaint.setColor(interiorColor);
        mInteriorPaint.setStyle(Paint.Style.FILL);
        mInteriorPaint.setAntiAlias(false);
        mInteriorPaint.setAlpha(255);

        mExteriorPaint = new Paint();
        mExteriorPaint.setTextSize(textSize);
        mExteriorPaint.setColor(exteriorColor);
        mExteriorPaint.setStyle(Paint.Style.FILL_AND_STROKE);
        mExteriorPaint.setStrokeWidth(textSize / 8);
        mExteriorPaint.setAntiAlias(false);
        mExteriorPaint.setAlpha(255);

        this.mTextSize = textSize;
    }

    public void setTypeFace(Typeface typeface) {
        mInteriorPaint.setTypeface(typeface);
        mExteriorPaint.setTypeface(typeface);
    }

    public void drawText(final Canvas canvas, final float posX, final float posY, final String text) {
        canvas.drawText(text, posX, posY, mExteriorPaint);
        canvas.drawText(text, posX, posY, mInteriorPaint);
    }

    public void drawLiness(Canvas canvas, final float posX, final float posY, Vector<String> lines) {
        int lineNum = 0;
        for (final String line : lines) {
            drawText(canvas, posX, posY - getTextSize() * (lines.size() - lineNum - 1), line);
            ++lineNum;
        }
    }

    public void setInteriorColor(final int color) {
        mInteriorPaint.setColor(color);
    }

    public void setExteriorColor(final int color) {
        mExteriorPaint.setColor(color);
    }

    public float getTextSize() {
        return mTextSize;
    }

    public void setAlpha(final int alpha) {
        mInteriorPaint.setAlpha(alpha);
        mExteriorPaint.setAlpha(alpha);
    }

    public void getTextBounds(final String line, final int index, final int count, final Rect lineBounds) {
        mInteriorPaint.getTextBounds(line, index, count, lineBounds);
    }

    public void setTextAlign(final Paint.Align align) {
        mInteriorPaint.setTextAlign(align);
        mExteriorPaint.setTextAlign(align);
    }
}
