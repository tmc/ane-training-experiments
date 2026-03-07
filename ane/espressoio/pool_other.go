//go:build !darwin

package espressoio

import "fmt"

type Pool struct{}

func Open(int, uint64) (*Pool, error) { return nil, fmt.Errorf("espressoio requires darwin") }
func (p *Pool) Close()                {}
func (p *Pool) Bytes() int            { return 0 }
func (p *Pool) Frames() uint64        { return 0 }

func (p *Pool) IOSurfaceForFrame(uint64) (uintptr, error) {
	return 0, fmt.Errorf("espressoio requires darwin")
}

func (p *Pool) SetExternalFrameStorage(uint64, uintptr) error {
	return fmt.Errorf("espressoio requires darwin")
}

func (p *Pool) RestoreInternalFrameStorage(uint64) error {
	return fmt.Errorf("espressoio requires darwin")
}

func (p *Pool) RestoreAllInternalStorage() error {
	return fmt.Errorf("espressoio requires darwin")
}

func (p *Pool) WriteFrame(uint64, []byte) error {
	return fmt.Errorf("espressoio requires darwin")
}

func (p *Pool) ReadFrame(uint64, []byte) error {
	return fmt.Errorf("espressoio requires darwin")
}
