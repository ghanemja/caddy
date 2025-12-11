// History Management for Mesh Processing Runs

/**
 * Save a run to history
 * @param {Object} runData - The run data to save
 * @param {string} runData.imageData - Base64 encoded image or image URL
 * @param {string} runData.instructions - User instructions/prompt
 * @param {string} runData.meshPath - Path to the output mesh
 * @param {Object} runData.parameters - Extracted parameters
 * @param {string} runData.category - Object category
 * @param {string} runData.timestamp - ISO timestamp
 * @param {boolean} runData.saved - Whether this run is saved (favorite)
 */
function saveRunToHistory(runData) {
    try {
        const history = getHistory();
        const runId = runData.id || `run_${Date.now()}`;
        
        const run = {
            id: runId,
            timestamp: runData.timestamp || new Date().toISOString(),
            imageData: runData.imageData,
            instructions: runData.instructions || '',
            meshPath: runData.meshPath,
            parameters: runData.parameters || [],
            category: runData.category || 'Unknown',
            saved: runData.saved || false,
        };
        
        // Add to history (most recent first)
        history.unshift(run);
        
        // Limit to 50 runs
        if (history.length > 50) {
            history.pop();
        }
        
        localStorage.setItem('mesh-processing-history', JSON.stringify(history));
        return runId;
    } catch (e) {
        console.error('[history] Error saving run:', e);
        return null;
    }
}

/**
 * Get all history runs
 * @returns {Array} Array of run objects
 */
function getHistory() {
    try {
        const historyJson = localStorage.getItem('mesh-processing-history');
        return historyJson ? JSON.parse(historyJson) : [];
    } catch (e) {
        console.error('[history] Error loading history:', e);
        return [];
    }
}

/**
 * Get a specific run by ID
 * @param {string} runId - The run ID
 * @returns {Object|null} The run object or null
 */
function getRunById(runId) {
    const history = getHistory();
    return history.find(run => run.id === runId) || null;
}

/**
 * Delete a run from history
 * @param {string} runId - The run ID to delete
 */
function deleteRun(runId) {
    try {
        const history = getHistory();
        const filtered = history.filter(run => run.id !== runId);
        localStorage.setItem('mesh-processing-history', JSON.stringify(filtered));
        return true;
    } catch (e) {
        console.error('[history] Error deleting run:', e);
        return false;
    }
}

/**
 * Toggle saved status of a run
 * @param {string} runId - The run ID
 * @param {boolean} saved - Whether to mark as saved
 */
function toggleRunSaved(runId, saved) {
    try {
        const history = getHistory();
        const run = history.find(r => r.id === runId);
        if (run) {
            run.saved = saved;
            localStorage.setItem('mesh-processing-history', JSON.stringify(history));
            return true;
        }
        return false;
    } catch (e) {
        console.error('[history] Error toggling saved status:', e);
        return false;
    }
}

/**
 * Clear all history (except saved runs if keepSaved is true)
 * @param {boolean} keepSaved - Whether to keep saved runs
 */
function clearHistory(keepSaved = true) {
    try {
        if (keepSaved) {
            const history = getHistory();
            const saved = history.filter(run => run.saved);
            localStorage.setItem('mesh-processing-history', JSON.stringify(saved));
        } else {
            localStorage.removeItem('mesh-processing-history');
        }
        return true;
    } catch (e) {
        console.error('[history] Error clearing history:', e);
        return false;
    }
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        saveRunToHistory,
        getHistory,
        getRunById,
        deleteRun,
        toggleRunSaved,
        clearHistory,
    };
}

