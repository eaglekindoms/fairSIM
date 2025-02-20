/*
This file is part of Free Analysis and Interactive Reconstruction
for Structured Illumination Microscopy (fairSIM).

fairSIM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

fairSIM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with fairSIM.  If not, see <http://www.gnu.org/licenses/>
*/

package org.fairsim.accel;

import org.fairsim.linalg.*;


/**
 * Vectors in native C code through JNI
 */
class AccelVectorCplx extends AbstractVectorCplx {

    /**
     * stores the C pointer
     */
    final long natData;
    boolean deviceNew = false, hostNew = false;

    /**
     * store the factory (and realm)
     */
    final protected AccelVectorFactory ourFactory;

    protected int ourCopyMode;

    /**
     * creates a new vector, allocates memory
     */
    AccelVectorCplx(AccelVectorFactory vf, int n) {
        super(n);
        ourFactory = vf;
        ourCopyMode = vf.defaultCopyMode();
        long nD = 0;
        try {
            nD = alloc(vf, n);
        } catch (Exception ex) {
        }
        natData = nD;
        if (natData == 0) {
            throw new java.lang.OutOfMemoryError("No memory for " +
                    "allocating native vector");
        }
    }

    /**
     * tells the native code to deallocate its memory
     */
    @Override
    protected void finalize() {
        dealloc(natData);
    }

    @Override
    public AccelVectorReal createAsReal(int n) {
        return ourFactory.createReal(n);
    }


    @Override
    public AccelVectorCplx duplicate() {
        AccelVectorCplx ret = ourFactory.createCplx(elemCount);
        ret.copy(this);
        return ret;
    }

    @Override
    public void readyBuffer() {
        if (deviceNew) {
            long buffer = 0;
            if (ourCopyMode == 2) {
                buffer = ourFactory.getNativeHostBuffer();
                if (buffer == 0)
                    throw new RuntimeException("null pointer (caught in JAVA)");
            }
            copyBuffer(natData, this.data, buffer, true, ourCopyMode);
        }
        deviceNew = false;
    }

    @Override
    public void syncBuffer() {
        long buffer = 0;
        if (ourCopyMode == 2) {
            buffer = ourFactory.getNativeHostBuffer();
            if (buffer == 0)
                throw new RuntimeException("null pointer (caught in JAVA)");
        }
        copyBuffer(natData, this.data, buffer, false, ourCopyMode);
        hostNew = false;
    }

    public void readyDevice() {
        if (hostNew)
            syncBuffer();
    }

    @Override
    public void makeCoherent() {
        if (hostNew && deviceNew)
            throw new RuntimeException("Changes occured to both device and host memory");
        if (hostNew)
            syncBuffer();
        if (deviceNew)
            readyBuffer();
    }


    // ------ mathematical functions ------

    /**
     * see if we have to call the 'super' function
     */
    boolean callSuper(Vec.Cplx v) {
        if (v == null) throw new NullPointerException("Input vector is null");
        if (!(v instanceof AccelVectorCplx))
            return true;

        Vec.failSize(this, v);
        return false;
    }

    /**
     * see if we have to call the 'super' function
     */
    boolean callSuper(Vec.Real v) {
        if (v == null) throw new NullPointerException("Input vector is null");
        if (!(v instanceof AccelVectorReal))
            return true;

        Vec.failSize(this, v);
        return false;
    }

    @Override
    public void add(Vec.Cplx... v) {
        if (v == null || v.length == 0)
            return;

        // Todo: currently if one vector is not our type,
        // everything goes through the slow function
        for (Vec.Cplx v1 : v)
            if (!(v1 instanceof AccelVectorCplx)) {
                super.add(v);
                //System.out.println("using fallback");
                return;
            }

        readyDevice();

        Vec.failSize(this, v);

        for (Vec.Cplx v1 : v) {
            ((AccelVectorCplx) v1).readyDevice();
            nativeAdd(this.natData, ((AccelVectorCplx) v1).natData, elemCount);
        }

        deviceNew = true;
    }

    @Override
    public void copy(Vec.Cplx in) {
        if (callSuper(in)) {
            super.copy(in);
            return;
        }
        nativeCOPYCPLX(this.natData, ((AccelVectorCplx) in).natData, elemCount);
        deviceNew = true;
    }

    @Override
    public void copy(Vec.Real in) {
        if (callSuper(in)) {
            super.copy(in);
            return;
        }
        nativeCOPYREAL(this.natData, ((AccelVectorCplx) in).natData, elemCount);
        deviceNew = true;
    }

    /*
    @Override
    public void addConst(float a) {
	readyDevice();
	nativeADDCONST( this.natData, elemCount, a, 0.f );
    } */

    @Override
    public void addConst(Cplx.Float a) {
        readyDevice();
        nativeADDCONST(this.natData, elemCount, a.re, a.im);
    }


    @Override
    public void axpy(float a, Vec.Cplx x) {
        if (callSuper(x)) {
            super.axpy(a, x);
            return;
        }
        readyDevice();
        ((AccelVectorCplx) x).readyDevice();
        nativeAXPY(a, 0, this.natData, ((AccelVectorCplx) x).natData, elemCount);
        deviceNew = true;
    }

    @Override
    public void axpy(Cplx.Float a, Vec.Cplx x) {
        if (callSuper(x)) {
            super.axpy(a, x);
            return;
        }
        readyDevice();
        ((AccelVectorCplx) x).readyDevice();
        nativeAXPY(a.re, a.im, this.natData, ((AccelVectorCplx) x).natData, elemCount);
        deviceNew = true;
    }

    @Override
    public void times(Vec.Cplx x, boolean conj) {
        if (callSuper(x)) {
            super.times(x, conj);
            return;
        }
        readyDevice();
        ((AccelVectorCplx) x).readyDevice();
        nativeTIMES(this.natData, ((AccelVectorCplx) x).natData, elemCount, conj);
        deviceNew = true;
    }

    @Override
    public void times(Vec.Real x) {
        if (callSuper(x)) {
            super.times(x);
            return;
        }
        readyDevice();
        ((AccelVectorReal) x).readyDevice();
        nativeTIMESREAL(this.natData, ((AccelVectorReal) x).natData, elemCount);
        deviceNew = true;
    }

    @Override
    public void elementwiseDivision(Vec.Real x) {
        /*
	if ( callSuper(x) ) {
	    super.times(x);
	    return;
	}
	readyDevice();
	((AccelVectorReal)x).readyDevice();
	nativeTIMESREAL( this.natData, ((AccelVectorReal)x).natData, elemCount );
	deviceNew = true;
        */
    }

    @Override
    public double norm2() {
        readyDevice();
        return nativeREDUCE(this.natData, elemCount, true);
    }

    @Override
    public void zero() {
        nativeZero(this.natData, elemCount);
        deviceNew = true;
    }

    @Override
    public void scal(Cplx.Float a) {
        readyDevice();
        nativeScal(this.natData, elemCount, a.re, a.im);
        deviceNew = true;
    }

    /*@Override
    public void scal(Cplx.Double a) {
	nativeScal( this.natData, elemCount, (float)a.re, (float)a.im );
    }*/

    @Override
    public void scal(float a) {
        readyDevice();
        nativeScal(this.natData, elemCount, a, 0.f);
        deviceNew = true;
    }

    // ------ direct native methods ------

    /**
     * set the vector to zero
     */
    public native void nativeZero(long addr, int elem);

    /**
     * multiply by scalar
     */
    public native void nativeScal(long addr, int elem, float re, float im);


    // ------ native methods ------

    /**
     * Allocate vector in native code
     */
    native long alloc(AccelVectorFactory vf, int n);

    /**
     * Disallocate vector
     */
    native void dealloc(long adrr);

    /**
     * sync to / from java code
     */
    native void copyBuffer(long addr, float[] jvec, long buffer, boolean toJava, int elem);

    /**
     * add vectors
     */
    native void nativeAdd(long addrThis, long addrOther, int len);

    /**
     * add constant float value
     */
    native void nativeADDCONST(long addrThis, int len, float re, float im);

    /**
     * axpy vectors
     */
    native void nativeAXPY(float re, float im, long addrThis, long addrOther, int len);

    /**
     * times vectors
     */
    native void nativeTIMES(long addrThis, long addrOther, int len, boolean conj);

    native void nativeTIMESREAL(long addrThis, long addrOther, int len);

    /**
     * norm2
     */
    native double nativeREDUCE(long addrThis, int len, boolean sqr);

    /**
     * copy vector
     */
    native void nativeCOPYREAL(long ptrOut, long ptrIn, int elem);

    native void nativeCOPYCPLX(long ptrOut, long ptrIn, int elem);
}


class AccelVectorCplx2d extends AccelVectorCplx implements Vec2d.Cplx {

    final int width, height;

    AccelVectorCplx2d(AccelVectorFactory vf, int w, int h) {
        super(vf, w * h);
        width = w;
        height = h;
    }

    // ------ Interface implementation ------

    @Override
    public int vectorWidth() {
        return width;
    }

    @Override
    public int vectorHeight() {
        return height;
    }

    @Override
    public void paste(Vec2d.Cplx in, int x, int y, boolean zero) {
        Vec2d.paste(in, this, 0, 0, in.vectorWidth(), in.vectorHeight(), x, y, zero);
    }

    @Override
    public AccelVectorCplx2d duplicate() {
        AccelVectorCplx2d ret = ourFactory.createCplx2D(width, height);
        ret.copy(this);
        return ret;
    }

    @Override
    public void set(int x, int y, Cplx.Float a) {
        //nativeSet(natData, x,y, width, (float)a.re, (float)a.im);
        readyBuffer();
        data[(x + y * width) * 2 + 0] = a.re;
        data[(x + y * width) * 2 + 1] = a.im;
        hostNew = true;
    }

    @Override
    public void set(int x, int y, Cplx.Double a) {
        readyBuffer();
        //nativeSet(natData, x,y, width, (float)a.re, (float)a.im);
        data[(x + y * width) * 2 + 0] = (float) a.re;
        data[(x + y * width) * 2 + 1] = (float) a.im;
        hostNew = true;
    }

    @Override
    public Cplx.Float get(int x, int y) {
        //float [] res = nativeGet( natData, x,y, width);
        //return new Cplx.Float( res[0], res[1]);
        readyBuffer();
        return new Cplx.Float(data[(x + y * width) * 2], data[(x + y * width) * 2 + 1]);
    }

    @Override
    public void pasteFreq(Vec2d.Cplx inV) {
        pasteFreq(inV, 0, 0);
    }

    @Override
    public void pasteFreq(Vec2d.Cplx inV, int xOffset, int yOffset) {

        final int wi = inV.vectorWidth();
        final int hi = inV.vectorHeight();
        final int wo = this.vectorWidth();
        final int ho = this.vectorHeight();

        if (!(inV instanceof AccelVectorCplx2d)) {
            // the slow way...

            final float[] out = this.vectorData();
            final float[] in = inV.vectorData();
            java.util.Arrays.fill(out, 0, elemCount * 2, 0);

            // loop output
            for (int y = 0; y < hi; y++)
                for (int x = 0; x < wi; x++) {
                    int xo = (x < wi / 2) ? (x) : (x + wo / 2);
                    int yo = (y < hi / 2) ? (y) : (y + ho / 2);
                    xo = (xo + xOffset + wo) % wo;
                    yo = (yo + yOffset + ho) % ho;
                    out[(xo + (wo * yo)) * 2 + 0] = in[(x + wi * y) * 2 + 0];
                    out[(xo + (wo * yo)) * 2 + 1] = in[(x + wi * y) * 2 + 1];
                }

            this.hostNew = true;

        } else {
            // the fast way
            nativePasteFreq(this.natData, wo, ho, ((AccelVectorCplx2d) inV).natData, wi, hi,
                    xOffset, yOffset);
            this.deviceNew = true;
        }
    }

    @Override
    public void setFrom16bitPixels(short[] in) {

        if (elemCount > AccelVectorFactory.nativeBufferSize / 8)
            throw new RuntimeException("Size exceeds buffer");

        long ptrbufHost = ourFactory.getNativeHostBuffer();
        long ptrbufDevice = ourFactory.getNativeDeviceBuffer();

        nativeCOPYSHORT(this.natData, ptrbufHost, ptrbufDevice, in, elemCount, ourCopyMode);
        deviceNew = true;

    }

    @Override
    public void fft2d(boolean inverse) {
        long fftPlan = FFTProvider.getOrCreateInstance(width, height);
        //System.out.println("starting with fftw: "+fftPlan );
        readyDevice();
        nativeFFT(fftPlan, natData, inverse);
        deviceNew = true;
        FFTProvider.returnInstance(width, height, fftPlan);
    }

    @Override
    public void fourierShift(final double kx, final double ky) {
        final int N = Vec2d.checkSquare(this);
        readyDevice();
        nativeFourierShift(this.natData, N, kx, ky);
        deviceNew = true;
    }

    @Override
    public void slice(Vec3d.Cplx in, int i) {
        throw new RuntimeException("Currently not implemented in AccelVectors");
    }

    @Override
    public void project(Vec3d.Cplx in, int start, int end) {
        throw new RuntimeException("Currently not implemented in AccelVectors");
    }

    @Override
    public void project(Vec3d.Cplx in) {
        throw new RuntimeException("Currently not implemented in AccelVectors");
    }


    // ------ Native methods ------

    native void nativeCOPYSHORT(long ptrOut, long hostBuffer, long deviceBuffer, short[] in,
                                int elem, int mode);

    native void nativeSet(long data, int x, int y, int width, float re, float im);

    native float[] nativeGet(long data, int x, int y, int width);

    native void nativeFFT(long fftPlan, long data, boolean inverse);

    native void nativeFourierShift(long data, int N, double kx, double ky);

    native void nativePasteFreq(long data, int wo, int ho, long indata, int wi, int hi,
                                int xOff, int yOff);


}

