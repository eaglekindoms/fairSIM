package org.fairsim.accel.mycuda;

/**
 * @author eagle
 * @date 2022/12/5 18:40
 */

import org.fairsim.accel.AccelVectorFactory;

public class TestCuFFT {
    public static void main(String[] args) {
        System.out.println(System.getProperty("java.library.path"));
    }

    /**
     * Allocate vector in native code
     */
    private long alloc(AccelVectorFactory vf, int n) {

        return 0l;
    }

}
