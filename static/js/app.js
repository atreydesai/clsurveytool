/**
 * CL Survey Tool - Frontend Logic
 */

// State
let state = {
    entries: { pending: [], saved: [] },
    constants: {},
    activeTab: 'pending',
    selectedEntryId: null,
    debounceTimer: null
};

// ============================================================================
// API Functions
// ============================================================================

async function api(endpoint, options = {}) {
    const response = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    return response.json();
}

async function loadEntries() {
    const data = await api('/entries');
    state.entries = { pending: data.pending || [], saved: data.saved || [] };
    state.constants = data.constants || {};
    renderEntryList();
    updateCounts();
}

async function importBibtex(bibtex) {
    const result = await api('/import', {
        method: 'POST',
        body: JSON.stringify({ bibtex })
    });
    if (result.entries) {
        await loadEntries();
    }
    return result;
}

async function saveEntry(entryId, data) {
    const result = await api(`/entries/${entryId}`, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
    showSaveIndicator();
    return result;
}

async function commitEntry(entryId) {
    const result = await api(`/entries/${entryId}/commit`, { method: 'POST' });
    if (!result.error) {
        await loadEntries();
        state.selectedEntryId = null;
        renderEditor();
    }
    return result;
}

async function deleteEntry(entryId) {
    const result = await api(`/entries/${entryId}`, { method: 'DELETE' });
    if (!result.error) {
        await loadEntries();
        if (state.selectedEntryId === entryId) {
            state.selectedEntryId = null;
            renderEditor();
        }
    }
    return result;
}

async function runAnalysis(notes) {
    return await api('/analyze', {
        method: 'POST',
        body: JSON.stringify({ notes })
    });
}

async function loadStats() {
    return await api('/stats');
}

// ============================================================================
// Rendering Functions
// ============================================================================

function updateCounts() {
    document.getElementById('pending-count').textContent = state.entries.pending.length;
    document.getElementById('saved-count').textContent = state.entries.saved.length;
}

function renderEntryList() {
    const list = document.getElementById('entry-list');
    const entries = state.activeTab === 'pending' ? state.entries.pending : state.entries.saved;

    if (entries.length === 0) {
        list.innerHTML = `<div class="entry-item"><p style="color: var(--text-muted);">No entries</p></div>`;
        return;
    }

    list.innerHTML = entries.map(entry => `
        <div class="entry-item ${state.selectedEntryId === entry.id ? 'active' : ''}" 
             data-id="${entry.id}">
            <div class="entry-title">${escapeHtml(entry.title || 'Untitled')}</div>
            <div class="entry-meta">${entry.year || '?'} · ${entry.journal || 'Unknown'}</div>
        </div>
    `).join('');

    // Add click handlers
    list.querySelectorAll('.entry-item').forEach(item => {
        item.addEventListener('click', () => {
            state.selectedEntryId = item.dataset.id;
            renderEntryList();
            renderEditor();
        });
    });
}

function renderEditor() {
    const editor = document.getElementById('editor');

    if (!state.selectedEntryId) {
        editor.innerHTML = `<div class="editor-placeholder"><p>Select an entry to edit</p></div>`;
        return;
    }

    const entry = [...state.entries.pending, ...state.entries.saved]
        .find(e => e.id === state.selectedEntryId);

    if (!entry) {
        editor.innerHTML = `<div class="editor-placeholder"><p>Entry not found</p></div>`;
        return;
    }

    const isPending = state.entries.pending.some(e => e.id === entry.id);
    const searchString = `${entry.title || ''} ${(entry.authors || [])[0] || ''}`;

    // Ensure minimum affiliations: 1 by default, 2 if multiple authors
    const authorCount = (entry.authors || []).length;
    const minAffiliations = authorCount > 1 ? 2 : 1;
    if (!entry.affiliations || entry.affiliations.length < minAffiliations) {
        entry.affiliations = entry.affiliations || [];
        while (entry.affiliations.length < minAffiliations) {
            entry.affiliations.push({ university: '', country: '', discipline: '' });
        }
    }

    editor.innerHTML = `
        <div class="editor-form">
            <div class="form-header">
                <h2 class="form-title">Edit Entry</h2>
                <div class="form-actions">
                    ${isPending ? `<button class="btn btn-success" id="btn-commit">Commit to Dataset</button>` : ''}
                    <button class="btn btn-danger btn-small" id="btn-delete">Delete</button>
                </div>
            </div>
            
            <div class="editor-columns">
                <!-- Left Column: Metadata -->
                <div class="editor-col-left">
                    <div class="form-row">
                        <div class="form-group" style="flex: 3;">
                            <label>Title</label>
                            <input type="text" id="field-title" value="${escapeHtml(entry.title || '')}">
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label>Year</label>
                            <input type="text" id="field-year" value="${escapeHtml(entry.year || '')}">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Journal/Venue</label>
                        <input type="text" id="field-journal" value="${escapeHtml(entry.journal || '')}">
                    </div>
                    
                    <div class="form-group">
                        <label>Authors (one per line)</label>
                        <textarea id="field-authors" style="min-height: 80px;">${(entry.authors || []).join('\n')}</textarea>
                    </div>
                    
                    <!-- Google Scholar Copy -->
                    <div class="form-group">
                        <label>Google Scholar Search</label>
                        <button class="copy-btn" id="btn-copy">📋 Copy: ${escapeHtml(searchString.substring(0, 40))}...</button>
                    </div>
                    
                    <!-- Analysis Notes -->
                    <div class="form-group">
                        <label>Analysis Notes</label>
                        <textarea id="field-notes" style="min-height: 150px;">${escapeHtml(entry.analysis_notes || '')}</textarea>
                        <button class="btn btn-primary" id="btn-analyze" style="margin-top: 0.5rem;">Run AI Analysis</button>
                    </div>
                </div>
                
                <!-- Right Column: Classifications -->
                <div class="editor-col-right">
                    <!-- Affiliations -->
                    <div class="form-group">
                        <label>Affiliations</label>
                        <table class="affiliations-table">
                            <thead>
                                <tr>
                                    <th>University</th>
                                    <th>Country</th>
                                    <th>Discipline</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody id="affiliations-body">
                                ${(entry.affiliations || []).map((aff, i) => `
                                    <tr data-index="${i}">
                                        <td><input type="text" class="aff-university" value="${escapeHtml(aff.university || '')}"></td>
                                        <td><input type="text" class="aff-country" value="${escapeHtml(aff.country || '')}"></td>
                                        <td><input type="text" class="aff-discipline" value="${escapeHtml(aff.discipline || '')}"></td>
                                        <td><button class="btn btn-danger btn-small aff-remove">×</button></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                        <button class="btn btn-small add-row-btn" id="btn-add-affiliation">+ Add Row</button>
                    </div>
                    
                    <!-- Species Categories -->
                    <div class="form-group">
                        <label>Species Categories</label>
                        <div class="tag-list" id="species-tags">
                            ${(entry.species_categories || []).map(cat => `
                                <span class="tag">${escapeHtml(cat)} <span class="tag-remove" data-value="${escapeHtml(cat)}">×</span></span>
                            `).join('')}
                        </div>
                        <select id="field-species-category" style="margin-top: 0.5rem;">
                            <option value="">Add category...</option>
                            ${(state.constants.species_categories || []).map(cat => `
                                <option value="${escapeHtml(cat)}" ${(entry.species_categories || []).includes(cat) ? 'disabled' : ''}>${escapeHtml(cat)}</option>
                            `).join('')}
                        </select>
                    </div>
                    
                    <!-- Specialized Species -->
                    <div class="form-group">
                        <label>Specialized Species (comma-separated)</label>
                        <input type="text" id="field-specialized-species" value="${(entry.specialized_species || []).join(', ')}">
                    </div>
                    
                    <!-- Computational Stages -->
                    <div class="form-group">
                        <label>Computational Stages</label>
                        <div class="tag-list" id="stages-tags">
                            ${(entry.computational_stages || []).map(stage => `
                                <span class="tag">${escapeHtml(stage)} <span class="tag-remove" data-value="${escapeHtml(stage)}">×</span></span>
                            `).join('')}
                        </div>
                        <select id="field-computational-stage" style="margin-top: 0.5rem;">
                            <option value="">Add stage...</option>
                            ${(state.constants.computational_stages || []).map(stage => `
                                <option value="${escapeHtml(stage)}" ${(entry.computational_stages || []).includes(stage) ? 'disabled' : ''}>${escapeHtml(stage)}</option>
                            `).join('')}
                        </select>
                    </div>
                    
                    <!-- Linguistic Features -->
                    <div class="form-group">
                        <label>Linguistic Features</label>
                        <div class="checkbox-grid-compact" id="features-grid">
                            ${(state.constants.linguistic_features || []).map((feat, i) => `
                                <label class="checkbox-item ${(entry.linguistic_features || []).includes(feat) ? 'checked' : ''}">
                                    <input type="checkbox" value="${escapeHtml(feat)}" ${(entry.linguistic_features || []).includes(feat) ? 'checked' : ''}>
                                    ${escapeHtml(feat)}
                                </label>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Attach event handlers
    attachEditorHandlers(entry);
}

function attachEditorHandlers(entry) {
    // Auto-save on input changes (debounced)
    const inputs = ['field-title', 'field-year', 'field-journal', 'field-authors', 'field-notes', 'field-specialized-species'];
    inputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', () => debounceAutoSave(entry));
        }
    });

    // Copy button
    document.getElementById('btn-copy')?.addEventListener('click', () => {
        const searchString = `${entry.title || ''} ${(entry.authors || [])[0] || ''}`;
        navigator.clipboard.writeText(searchString);
        document.getElementById('btn-copy').textContent = '✅ Copied!';
        setTimeout(() => renderEditor(), 1500);
    });

    // Commit button
    document.getElementById('btn-commit')?.addEventListener('click', async () => {
        await commitEntry(entry.id);
    });

    // Delete button
    document.getElementById('btn-delete')?.addEventListener('click', async () => {
        if (confirm('Delete this entry?')) {
            await deleteEntry(entry.id);
        }
    });

    // AI Analysis button
    document.getElementById('btn-analyze')?.addEventListener('click', async () => {
        const notes = document.getElementById('field-notes').value;
        if (!notes) {
            alert('Add analysis notes first');
            return;
        }
        const btn = document.getElementById('btn-analyze');
        btn.disabled = true;
        btn.textContent = 'Analyzing...';

        const result = await runAnalysis(notes);
        if (result.error) {
            alert('Error: ' + result.error);
        } else {
            // Apply results
            entry.linguistic_features = result.linguistic_features || [];
            entry.species_categories = [...new Set(result.species_categories || [])];
            entry.specialized_species = result.specialized_species || [];
            entry.computational_stages = result.computational_stages || [];
            await saveEntry(entry.id, entry);
            renderEditor();
        }
        btn.disabled = false;
        btn.textContent = 'Run AI Analysis';
    });

    // Add affiliation row
    document.getElementById('btn-add-affiliation')?.addEventListener('click', () => {
        if (!entry.affiliations) entry.affiliations = [];
        entry.affiliations.push({ university: '', country: '', discipline: '' });
        renderEditor();
    });

    // Remove affiliation row
    document.querySelectorAll('.aff-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const row = e.target.closest('tr');
            const index = parseInt(row.dataset.index);
            entry.affiliations.splice(index, 1);
            collectFormData(entry);
            saveEntry(entry.id, entry);
            renderEditor();
        });
    });

    // Affiliation input changes
    document.querySelectorAll('.affiliations-table input').forEach(input => {
        input.addEventListener('input', () => debounceAutoSave(entry));
    });

    // Species category select
    document.getElementById('field-species-category')?.addEventListener('change', (e) => {
        if (e.target.value) {
            if (!entry.species_categories) entry.species_categories = [];
            if (!entry.species_categories.includes(e.target.value)) {
                entry.species_categories.push(e.target.value);
                saveEntry(entry.id, { species_categories: entry.species_categories });
                renderEditor();
            }
        }
    });

    // Remove species category
    document.querySelectorAll('#species-tags .tag-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            const value = btn.dataset.value;
            entry.species_categories = (entry.species_categories || []).filter(c => c !== value);
            saveEntry(entry.id, { species_categories: entry.species_categories });
            renderEditor();
        });
    });

    // Computational stage select
    document.getElementById('field-computational-stage')?.addEventListener('change', (e) => {
        if (e.target.value) {
            if (!entry.computational_stages) entry.computational_stages = [];
            if (!entry.computational_stages.includes(e.target.value)) {
                entry.computational_stages.push(e.target.value);
                saveEntry(entry.id, { computational_stages: entry.computational_stages });
                renderEditor();
            }
        }
    });

    // Remove computational stage
    document.querySelectorAll('#stages-tags .tag-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            const value = btn.dataset.value;
            entry.computational_stages = (entry.computational_stages || []).filter(s => s !== value);
            saveEntry(entry.id, { computational_stages: entry.computational_stages });
            renderEditor();
        });
    });

    // Linguistic features checkboxes
    document.querySelectorAll('#features-grid input').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const checked = Array.from(document.querySelectorAll('#features-grid input:checked'))
                .map(cb => cb.value);
            entry.linguistic_features = checked;
            saveEntry(entry.id, { linguistic_features: checked });
        });
    });
}

function collectFormData(entry) {
    // Collect all form data into entry
    entry.title = document.getElementById('field-title')?.value || '';
    entry.year = document.getElementById('field-year')?.value || '';
    entry.journal = document.getElementById('field-journal')?.value || '';
    entry.authors = (document.getElementById('field-authors')?.value || '').split('\n').filter(a => a.trim());
    entry.analysis_notes = document.getElementById('field-notes')?.value || '';
    entry.specialized_species = (document.getElementById('field-specialized-species')?.value || '')
        .split(',').map(s => s.trim()).filter(s => s);

    // Affiliations
    entry.affiliations = [];
    document.querySelectorAll('#affiliations-body tr').forEach(row => {
        const aff = {
            university: row.querySelector('.aff-university')?.value || '',
            country: row.querySelector('.aff-country')?.value || '',
            discipline: row.querySelector('.aff-discipline')?.value || ''
        };
        if (aff.university || aff.country || aff.discipline) {
            entry.affiliations.push(aff);
        }
    });
}

function debounceAutoSave(entry) {
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
        collectFormData(entry);
        saveEntry(entry.id, entry);
    }, 500);
}

function showSaveIndicator() {
    let indicator = document.querySelector('.saving-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.className = 'saving-indicator';
        indicator.textContent = '✓ Saved';
        document.body.appendChild(indicator);
    }
    indicator.classList.add('show');
    setTimeout(() => indicator.classList.remove('show'), 1500);
}

function renderAnalytics(stats) {
    const grid = document.getElementById('stats-grid');

    if (stats.empty) {
        grid.innerHTML = `<p style="color: var(--text-muted);">No data yet. Commit some entries first.</p>`;
        return;
    }

    grid.innerHTML = `
        <div class="stat-card">
            <h3>Total Papers</h3>
            <div class="stat-number">${stats.total || 0}</div>
        </div>
        
        <div class="stat-card">
            <h3>By Year</h3>
            <div class="stat-bars">
                ${Object.entries(stats.years || {}).sort((a, b) => b[0] - a[0]).slice(0, 8).map(([year, count]) => `
                    <div class="stat-bar">
                        <span class="stat-bar-label">${year}</span>
                        <div class="stat-bar-fill"><div class="stat-bar-fill-inner" style="width: ${(count / stats.total) * 100}%"></div></div>
                        <span class="stat-bar-value">${count}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="stat-card">
            <h3>Species Categories</h3>
            <div class="stat-bars">
                ${Object.entries(stats.species || {}).sort((a, b) => b[1] - a[1]).map(([cat, count]) => `
                    <div class="stat-bar">
                        <span class="stat-bar-label">${cat}</span>
                        <div class="stat-bar-fill"><div class="stat-bar-fill-inner" style="width: ${(count / stats.total) * 100}%"></div></div>
                        <span class="stat-bar-value">${count}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="stat-card">
            <h3>Computational Stages</h3>
            <div class="stat-bars">
                ${Object.entries(stats.stages || {}).sort((a, b) => b[1] - a[1]).map(([stage, count]) => `
                    <div class="stat-bar">
                        <span class="stat-bar-label">${stage}</span>
                        <div class="stat-bar-fill"><div class="stat-bar-fill-inner" style="width: ${(count / stats.total) * 100}%"></div></div>
                        <span class="stat-bar-value">${count}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// ============================================================================
// Event Handlers
// ============================================================================

function setupEventHandlers() {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.activeTab = tab.dataset.tab;
            state.selectedEntryId = null;
            renderEntryList();
            renderEditor();
        });
    });

    // Import modal
    document.getElementById('btn-import').addEventListener('click', () => {
        document.getElementById('import-modal').style.display = 'flex';
    });

    document.getElementById('modal-close').addEventListener('click', () => {
        document.getElementById('import-modal').style.display = 'none';
    });

    document.getElementById('btn-parse').addEventListener('click', async () => {
        const bibtex = document.getElementById('bibtex-input').value;
        if (!bibtex) {
            alert('Paste BibTeX content first');
            return;
        }
        const result = await importBibtex(bibtex);
        if (result.error) {
            alert('Error: ' + result.error);
        } else {
            document.getElementById('import-modal').style.display = 'none';
            document.getElementById('bibtex-input').value = '';
        }
    });

    // Analytics toggle
    document.getElementById('btn-analytics').addEventListener('click', async () => {
        const editor = document.getElementById('editor');
        const analytics = document.getElementById('analytics');

        if (analytics.style.display === 'none') {
            editor.style.display = 'none';
            analytics.style.display = 'block';
            const stats = await loadStats();
            renderAnalytics(stats);
            document.getElementById('btn-analytics').classList.add('active');
        } else {
            editor.style.display = 'block';
            analytics.style.display = 'none';
            document.getElementById('btn-analytics').classList.remove('active');
        }
    });

    // Close modal on outside click
    document.getElementById('import-modal').addEventListener('click', (e) => {
        if (e.target.id === 'import-modal') {
            document.getElementById('import-modal').style.display = 'none';
        }
    });
}

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventHandlers();
    loadEntries();
});
