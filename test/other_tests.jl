
p = rand(MRP)
pval = SVector(p)
@test skew(p) == skew(pval)

@test vee(skew(pval)) == pval

p2 = rand(MRP)
@test p ⊖ p2 == SVector(p2\p)
