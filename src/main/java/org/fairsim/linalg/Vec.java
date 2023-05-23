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

package org.fairsim.linalg;


/**
 * Interface definition for linear algebra (vectors).
 * Available functions should be somewhat universal, but
 * of course cater to SIM reconstructions (FFTs, shifts, etc).
 * There is some convention to how the code works, which should
 * be followed:
 *
 * <ul>
 * <li>Vectors are nested interfaces in the Vec, Vec2d, Vec3d class
 * <li>This is because Vec.Cplx, Vec2d.Real seem most readable to me
 * <li>Input parameters are always read-only, e.g. a.sub(b)
 *  will only change a, never b. Think "const" if this would be C++
 * <li>Memory allocation only happens through methods directly in
 * Vec, Vec2d ... class, NOT in the nested interfaces. E.g. do NOT
 * implement Vec.Cplx.duplicate(), but Vec.duplicate( Vec.Cplx )
 * <li>All memory allocation goes through the vector factory
 * </ul>
 */
public final class Vec {

    // ================================================================

    /**
     * private constructor, so this class only has static methods
     */
    private Vec() {
        throw new AssertionError();
    }

    /**
     * Vector factory to use
     */
    static VectorFactory vf = BasicVector.getFactory();

    /**
     * Set the factory used for obtaining vectors through
     * the static 'create' functions.
     */
    public static void setVectorFactory(VectorFactory invf) {
        if (vf == null)
            throw new java.lang.NullPointerException("vf was null, not set");
        vf = invf;
    }

    /**
     * Creates a real-valued vector of size n.
     */
    public static Real createReal(int n) {
        return vf.createReal(n);
    }

    /**
     * Creates a complex-valued vector of size n.
     */
    public static Cplx createCplx(int n) {
        return vf.createCplx(n);
    }

    /**
     * Creates a k-element array of real-valued vector of size n.
     */
    public static Real[] createArrayReal(int k, int n) {
        Real[] ret = new Real[k];
        for (int i = 0; i < k; i++)
            ret[i] = vf.createReal(n);
        return ret;
    }

    /**
     * Creates a k-element array of complex-valued vector of size n.
     */
    public static Cplx[] createArrayCplx(int k, int n) {
        Cplx[] ret = new Cplx[k];
        for (int i = 0; i < k; i++)
            ret[i] = vf.createCplx(n);
        return ret;
    }

    /**
     * Finish whatever parallel / concurrent process is running (for timing CUDA, etc.).
     * This calls {VectorFactory#syncConcurrent} of the current vector factory.
     */
    public static void syncConcurrent() {
        vf.syncConcurrent();
    }

    /**
     * Return the basic (non-accelerated) vector factory
     */
    public static VectorFactory getBasicVectorFactory() {
        return BasicVector.getFactory();
    }
    // ================================================================

    /**
     * Throw runtime exception if not all vectors are of same size.
     */
    public static int failSize(Vec.Real v0, Vec.Real... v) {
        if (v.length < 1)
            return v0.vectorSize();

        int len = v0.vectorSize();
        for (int i = 0; i < v.length; i++)
            if (v[i].vectorSize() != len)
                throw new RuntimeException("Vector len mismatch"
                        + i + " " + len + " " + v[i].vectorSize());
        return len;
    }

    /**
     * Throw runtime exception if not all vectors are of same size.
     */
    public static int failSize(Vec.Cplx v0, Vec.Cplx... v) {
        if (v.length < 1)
            return v0.vectorSize();

        int len = v0.vectorSize();
        for (int i = 0; i < v.length; i++)
            if (v[i].vectorSize() != len)
                throw new RuntimeException("Vector len mismatch"
                        + i + " " + len + " " + v[i].vectorSize());
        return len;
    }

    /**
     * Throw runtime exception if not all vectors are of same size.
     */
    public static int failSize(Vec.Cplx v0, Vec.Real... v) {
        if (v.length < 1)
            return v0.vectorSize();

        int len = v0.vectorSize();
        for (int i = 0; i < v.length; i++)
            if (v[i].vectorSize() != len)
                throw new RuntimeException("Vector len mismatch"
                        + i + " " + len + " " + v[i].vectorSize());
        return len;
    }

    /**
     * Throw runtime exception if not all vectors are of same size.
     */
    public static int failSize(Vec.Real v0, Vec.Cplx... v) {
        if (v.length < 1)
            return v0.vectorSize();

        int len = v0.vectorSize();
        for (int i = 0; i < v.length; i++)
            if (v[i].vectorSize() != len)
                throw new RuntimeException("Vector len mismatch"
                        + i + " " + len + " " + v[i].vectorSize());
        return len;
    }


    // ================================================================

    /**
     * A real-valued vector, base-type float
     */
    public interface Real {

        /**
         * Return a copy/duplicate of this vector
         */
		Real duplicate();

        /**
         * Number of elements in the vector
         */
		int vectorSize();

        /**
         * Access to the internal vector array.
         * If the vector is backed by external memory (GPU, etc.) it
         * will be synced before the buffer is returned. If the buffer
         * is written to, call {@link #syncBuffer} after the write.
         * Also, any call to methods in this vector (add, mult, etc.)
         * may invalidate the buffer.
         */
		float[] vectorData();

        /**
         * Sync content from the last reference returned by
         * {@link #vectorData}, invalidating the reference.
         * After a sync call, the buffer has to be re-obtained through
         * vectorData.
         */
		void syncBuffer();


        /**
         * Make all buffers (CPU, GPU, ...) coherent. This call ONLY influence
         * WHEN buffers are syncronized (for timing, etc.), the program will run
         * correctly without explicitly calling makeCoherent.
         */
		void makeCoherent();


        /**
         * Get the n'th element. Note: Depending on implementation,
         * when running a loop over the full vector, obtaining the vector
         * buffer directly might be faster
         */
		float get(int i);

        /**
         * Set the n'th element. Note: Depending on implementation,
         * when running a loop over the full vector, obtaining the vector
         * buffer directly and syncing it afterwards might be faster.
         */
		void set(int i, float x);


        // --- copy functions ---

        /**
         * Copy the content of 'in' into this vector
         */
		void copy(Real in);

        /**
         * Copy the real/imag values of 'in' into this vector
         */
		void copy(Cplx in, boolean imag);

        /**
         * Copy the real values of 'in' into this vector
         */
		void copy(Cplx in);

        /**
         * Copy the magnitudes of elements in 'in' into this vector
         */
		void copyMagnitude(Cplx in);

        /**
         * Copy the phases of elements in 'in' into this vector
         */
		void copyPhase(Cplx in);

        /**
         * Set all elements to zero
         */
		void zero();

        /**
         * Add all input vectors to this vector
         */
		void add(Real... in);

        /**
         * Computes this += a * x
         */
		void axpy(float a, Vec.Real x);

        /**
         * Add a constant value to each component
         */
		void addConst(float a);

        /**
         * Multiply by scalar, ie this *= a
         */
		void scal(float a);

        /**
         * Return the dot product < this, x >
         */
		double dot(Real x);

        /**
         * Return the squared norm <this, this>
         */
		double norm2();

        /**
         * Compute the elemnt-wise multiplication this = this.*x
         */
		void times(Vec.Real x);

        /**
         * Return the sum of all vector elements
         */
		double sumElements();

        /**
         * Normalize the vector to 0..1
         */
		void normalize();

        /**
         * Normalize the vector to vmin..vmax
         */
		void normalize(float vmin, float vmax);


        /**
         * Set every element to 1/element
         */
		void reciproc();

        /**
         * Compute y = y + x^2
         */
		void addSqr(Vec.Real xIn);

        /**
         * Output the first 10 vector elements for debugging
         */
		String first10Elem();


        /**
         * Return the index of the n largest elements
         */
		int[] nLargestIdx(int n);


        /**
         * Computes the average of all elements
         */
		double avr();

        /**
         * Computes the median of all elements
         */
		double median();

        /**
         * Returns the minimum value
         */
		float min();

        /**
         * Return the maximum value
         */
		float max();


    }


    // ================================================================


    /**
     * A complex-valued vector, base-type float
     */
    public interface Cplx {


        /**
         * called after writing to the buffer. Typically,
         * copies data back to the accelarator.
         */
		void syncBuffer();

        /**
         * Make all buffers (CPU, GPU, ...) coherent. This call ONLY influence
         * WHEN buffers are syncronized (for timing, etc.), the program will run
         * correctly without explicitly calling makeCoherent.
         */
		void makeCoherent();

        /**
         * Return a deep copy/duplicate of this vector. Implementations
         * should typically return their specific type here.
         */
		Cplx duplicate();

        /**
         * Get the n'th element. Note: Depending on implementation,
         * when running a loop over the full vector, obtaining the vector
         * buffer directly might be faster
         */
		org.fairsim.linalg.Cplx.Float get(int n);

        /**
         * Set the n'th element. Note: Depending on implementation,
         * when running a loop over the full vector, obtaining the vector
         * buffer directly and syncing it afterwards might be faster.
         */
		void set(int n, org.fairsim.linalg.Cplx.Float v);

        /**
         * Set the n'th element. Note: Depending on implementation,
         * when running a loop over the full vector, obtaining the vector
         * buffer directly and syncing it afterwards might be faster.
         */
		void set(int n, org.fairsim.linalg.Cplx.Double v);

        // --- creating new vectors ---


        /**
         * Return a copy of the real elements in this vector
         */
        //@Deprecated // TODO: re-deprecate all these 'duplicate' functions!
		Real duplicateReal();

        /**
         * Return a copy of the imaginary elements in this vector
         */
        //@Deprecated
		Real duplicateImag();

        /**
         * Return a copy, containing the magnitudes of elements in this vector
         */
        //@Deprecated
		Real duplicateMagnitude();

        /**
         * Return a copy, containing the phases of elements in this vector
         */
        //@Deprecated
		Real duplicatePhase();

        // --- arith. functions ---

        /**
         * Copy the content of 'in' into this vector
         */
		void copy(Cplx in);

        /**
         * Copy the content of 'in' into this vector
         */
		void copy(Real in);

        /**
         * Set all elements to zero
         */
		void zero();

        /**
         * Add all input vectors to this vector
         */
		void add(Cplx... in);

        /**
         * Computes this += a * x
         */
		void axpy(float a, Vec.Cplx x);

        /**
         * Computes this += a * x
         */
		void axpy(org.fairsim.linalg.Cplx.Float a, Vec.Cplx x);

        /**
         * Set every element to 1/element
         */
		void reciproc();

        /**
         * Add a constant value to each component
         */
		void addConst(org.fairsim.linalg.Cplx.Float a);

        /**
         * Multiply by scalar, ie this *= a
         */
		void scal(float a);

        /**
         * Scale by 'in', ie this *= in
         */
		void scal(org.fairsim.linalg.Cplx.Float in);

        /**
         * Return the squared norm <this, this>
         */
		double norm2();

        /**
         * Complex conjugate every element of this vector
         */
		void conj();

        /**
         * Compute the dot product < this^, y >
         */
		org.fairsim.linalg.Cplx.Double dot(Vec.Cplx yIn);

        /**
         * Compute element-wise multiplication this = this.*in.
         */
		void times(Cplx in);

        /**
         * Compute element-wise multiplication this = this.*conj(in)
         */
		void timesConj(Cplx in);

        /**
         * Compute element-wise multiplication this = this.*in,
         * conjugated 'in' if 'conj' is true
         */
		void times(Cplx in, final boolean conj);

        /**
         * Compute element-wise multiplication this = this.*in
         */
		void times(Real in);

        /**
         * Return the sum of all vector elements
         */
		org.fairsim.linalg.Cplx.Double sumElements();

        /**
         * Compute this += x^2
         */
		void addSqr(Vec.Cplx xIn);

        /**
         * Access to the internal vector array
         */
		float[] vectorData();

        /**
         * Vector size
         */
		int vectorSize();

        /**
         * Output the first 10 vector elements for debugging
         */
		String first10Elem();

    }

}

