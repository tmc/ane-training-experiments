module github.com/maderix/ANE

go 1.25.2

require (
	github.com/ebitengine/purego v0.10.0
	github.com/tmc/apple v0.3.3-0.20260315071352-5021d6701d1a
	github.com/tmc/mlx-go v0.0.0
)

require github.com/tmc/aneperf v0.0.0-20260315013400-e1bd670fe867 // indirect

replace github.com/tmc/apple => /Users/tmc/go/src/github.com/tmc/apple

replace github.com/tmc/mlx-go => /Users/tmc/go/src/github.com/tmc/mlx-go

replace github.com/ebitengine/purego => github.com/tmc/purego v0.10.0-alpha.2.0.20260130081008-0b23e28544a2
