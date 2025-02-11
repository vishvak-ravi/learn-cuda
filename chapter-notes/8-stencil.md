# stencil
## basics
- you been doing this (Newton's, RK4, cubic splines, almost every numerical algo ever)
- ... hardly different in higher dimensions
- akin to conv but requires higher precision (more memory!)
    - also note 2d 5pt stencil is analogous to convolution (3 x 3) without borders!
- since fewer input items per output item -> lower ops/byte moved than with conv
    - discrepancy prounounced when order increases
        - even more prounounced if dimension increases
- recall, halo elements are are less reused than interior ones...
- practically, input tiles are cubes of T < 8 since 8^3 = 512, limiting with SM's not supporting much higher
    - small T carries lots more halo elements, compared to convolution with reasonable numbers more than half of input are halo