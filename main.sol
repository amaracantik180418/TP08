// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title TensorProxima_08
/// @notice On-chain registry for AI training run metadata: epochs, loss snapshots, and checkpoint hashes.
/// @dev Used by training pipelines to anchor run identifiers and metrics without storing full datasets.
///      Proxima anchor suite; indexers expect configHash = keccak256(abi.encode(hyperparams)).

contract TensorProxima_08 {

    // -------------------------------------------------------------------------
    // EVENTS
    // -------------------------------------------------------------------------

    event RunRegistered(
        bytes32 indexed runId,
        address indexed submitter,
        uint16 epochCount,
        bytes32 configHash,
        uint256 anchoredAt
    );
    event EpochRecorded(
        bytes32 indexed runId,
        uint32 indexed epochIndex,
        uint256 lossScaled,
        bytes32 gradientRoot,
        uint256 recordedAt
    );
    event CheckpointAnchored(
        bytes32 indexed runId,
        uint32 checkpointIndex,
        bytes32 stateHash,
        uint256 anchoredAt
    );
    event CuratorUpdated(address indexed previousCurator, address indexed newCurator);
    event FeeCollectorUpdated(address indexed previousCollector, address indexed newCollector);
    event AnchorFeeSet(uint256 previousFeeWei, uint256 newFeeWei);
    event RunArchived(bytes32 indexed runId, address indexed archivedBy, uint256 atBlock);
    event TreasuryPull(address indexed to, uint256 amountWei, uint256 atBlock);

    // -------------------------------------------------------------------------
    // ERRORS
    // -------------------------------------------------------------------------

    error TP08_NotCurator();
    error TP08_ZeroAddress();
    error TP08_RunNotFound();
    error TP08_RunAlreadyArchived();
    error TP08_EpochIndexOutOfRange();
    error TP08_InvalidRunId();
    error TP08_InvalidConfigHash();
