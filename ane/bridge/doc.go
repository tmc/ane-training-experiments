// Package bridge loads libane_bridge.dylib and exposes its C entry points.
//
// The package is optional. Callers can use it to drive shared-event and
// daemon-backed ANE flows from Go without hardcoding dylib locations.
package bridge
