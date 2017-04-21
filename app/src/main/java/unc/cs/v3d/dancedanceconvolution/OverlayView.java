package unc.cs.v3d.dancedanceconvolution;

import android.content.Context;
import android.graphics.Canvas;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by kewang on 4/20/17.
 */

public class OverlayView extends View {
    public OverlayView(Context context) {
        super(context);
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public interface DrawCallback {
        public void drawCallback(final Canvas canvas);
    }

    public void addCallback(final DrawCallback callback) {
        mDrawCallbacks.add(callback);
    }

    private final List<DrawCallback> mDrawCallbacks = new LinkedList<>();


    @Override
    public void draw(Canvas canvas) {
        for (final DrawCallback callback : mDrawCallbacks) {
            callback.drawCallback(canvas);
        }
        super.draw(canvas);
    }
}
