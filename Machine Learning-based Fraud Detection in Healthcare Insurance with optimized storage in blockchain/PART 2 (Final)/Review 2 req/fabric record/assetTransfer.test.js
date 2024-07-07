/*
 * Copyright IBM Corp. All Rights Reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
*/

'use strict';
const sinon = require('sinon');
const chai = require('chai');
const sinonChai = require('sinon-chai');
const expect = chai.expect;

const { Context } = require('fabric-contract-api');
const { ChaincodeStub } = require('fabric-shim');

const AssetTransfer = require('../lib/assetTransfer.js');

let assert = sinon.assert;
chai.use(sinonChai);

describe('Asset Transfer Basic Tests', () => {
    let transactionContext, chaincodeStub, asset;
    beforeEach(() => {
        transactionContext = new Context();

        chaincodeStub = sinon.createStubInstance(ChaincodeStub);
        transactionContext.setChaincodeStub(chaincodeStub);

        chaincodeStub.putState.callsFake((key, value) => {
            if (!chaincodeStub.states) {
                chaincodeStub.states = {};
            }
            chaincodeStub.states[key] = value;
        });

        chaincodeStub.getState.callsFake(async (key) => {
            let ret;
            if (chaincodeStub.states) {
                ret = chaincodeStub.states[key];
            }
            return Promise.resolve(ret);
        });

        chaincodeStub.deleteState.callsFake(async (key) => {
            if (chaincodeStub.states) {
                delete chaincodeStub.states[key];
            }
            return Promise.resolve(key);
        });

        chaincodeStub.getStateByRange.callsFake(async () => {
            function* internalGetStateByRange() {
                if (chaincodeStub.states) {
                    // Shallow copy
                    const copied = Object.assign({}, chaincodeStub.states);

                    for (let key in copied) {
                        yield {value: copied[key]};
                    }
                }
            }

            return Promise.resolve(internalGetStateByRange());
        });

        asset = {
            ID: 'record1',
            State: '39',
            County: 230,
            ChronicCond_Alzheimer: '1',
            ClmAdmitDiagnosisCode: 0,
        };
    });

    describe('Test InitLedger', () => {
        it('should return error on InitLedger', async () => {
            chaincodeStub.putState.rejects('failed inserting key');
            let assetTransfer = new AssetTransfer();
            try {
                await assetTransfer.InitLedger(transactionContext);
                assert.fail('InitLedger should have failed');
            } catch (err) {
                expect(err.name).to.equal('failed inserting key');
            }
        });

        it('should return success on InitLedger', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.InitLedger(transactionContext);
            let ret = JSON.parse((await chaincodeStub.getState('record1')).toString());
            expect(ret).to.eql(Object.assign({docType: 'asset'}, asset));
        });
    });

    describe('Test CreateAsset', () => {
        it('should return error on CreateAsset', async () => {
            chaincodeStub.putState.rejects('failed inserting key');

            let assetTransfer = new AssetTransfer();
            try {
                await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);
                assert.fail('CreateAsset should have failed');
            } catch(err) {
                expect(err.name).to.equal('failed inserting key');
            }
        });

        it('should return success on CreateAsset', async () => {
            let assetTransfer = new AssetTransfer();

            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            let ret = JSON.parse((await chaincodeStub.getState(asset.ID)).toString());
            expect(ret).to.eql(asset);
        });
    });

    describe('Test ReadAsset', () => {
        it('should return error on ReadAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            try {
                await assetTransfer.ReadAsset(transactionContext, 'record2');
                assert.fail('ReadAsset should have failed');
            } catch (err) {
                expect(err.message).to.equal('The record with id record2 does not exist');
            }
        });

        it('should return success on ReadAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            let ret = JSON.parse(await chaincodeStub.getState(asset.ID));
            expect(ret).to.eql(asset);
        });
    });

    describe('Test UpdateAsset', () => {
        it('should return error on UpdateAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            try {
                await assetTransfer.UpdateAsset(transactionContext, 'record2', '41', 620, '1', 0);
                assert.fail('UpdateAsset should have failed');
            } catch (err) {
                expect(err.message).to.equal('The record with id record2 does not exist');
            }
        });

        it('should return success on UpdateAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            await assetTransfer.UpdateAsset(transactionContext, 'record1', '93', 320, '0', 1);
            let ret = JSON.parse(await chaincodeStub.getState(asset.ID));
            let expected = {
                ID: 'record1',
                State: '93',
                County: 320,
                ChronicCond_Alzheimer: '0',
                ClmAdmitDiagnosisCode: 1
            };
            expect(ret).to.eql(expected);
        });
    });

    describe('Test DeleteAsset', () => {
        it('should return error on DeleteAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            try {
                await assetTransfer.DeleteAsset(transactionContext, 'record2');
                assert.fail('DeleteAsset should have failed');
            } catch (err) {
                expect(err.message).to.equal('The record with record2 does not exist');
            }
        });

        it('should return success on DeleteAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            await assetTransfer.DeleteAsset(transactionContext, asset.ID);
            let ret = await chaincodeStub.getState(asset.ID);
            expect(ret).to.equal(undefined);
        });
    });

    describe('Test TransferAsset', () => {
        it('should return error on TransferAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            try {
                await assetTransfer.TransferAsset(transactionContext, 'record2', '1');
                assert.fail('DeleteAsset should have failed');
            } catch (err) {
                expect(err.message).to.equal('The asset asset2 does not exist');
            }
        });

        it('should return success on TransferAsset', async () => {
            let assetTransfer = new AssetTransfer();
            await assetTransfer.CreateAsset(transactionContext, asset.ID, asset.State, asset.County, asset.ChronicCond_Alzheimer, asset.ClmAdmitDiagnosisCode);

            await assetTransfer.TransferAsset(transactionContext, asset.ID, '1');
            let ret = JSON.parse((await chaincodeStub.getState(asset.ID)).toString());
            expect(ret).to.eql(Object.assign({}, asset, {ChronicCond_Alzheimer: '1'}));
        });
    });

    describe('Test GetAllAssets', () => {
        it('should return success on GetAllAssets', async () => {
            let assetTransfer = new AssetTransfer();

            await assetTransfer.CreateAsset(transactionContext, 'asset1', 'blue', 5, 'Robert', 100);
            await assetTransfer.CreateAsset(transactionContext, 'asset2', 'orange', 10, 'Paul', 200);
            await assetTransfer.CreateAsset(transactionContext, 'asset3', 'red', 15, 'Troy', 300);
            await assetTransfer.CreateAsset(transactionContext, 'asset4', 'pink', 20, 'Van', 400);

            let ret = await assetTransfer.GetAllAssets(transactionContext);
            ret = JSON.parse(ret);
            expect(ret.length).to.equal(4);

            let expected = [
                {Record: {ID: 'asset1', Color: 'blue', Size: 5, Owner: 'Robert', AppraisedValue: 100}},
                {Record: {ID: 'asset2', Color: 'orange', Size: 10, Owner: 'Paul', AppraisedValue: 200}},
                {Record: {ID: 'asset3', Color: 'red', Size: 15, Owner: 'Troy', AppraisedValue: 300}},
                {Record: {ID: 'asset4', Color: 'pink', Size: 20, Owner: 'Van', AppraisedValue: 400}}
            ];

            expect(ret).to.eql(expected);
        });

        it('should return success on GetAllAssets for non JSON value', async () => {
            let assetTransfer = new AssetTransfer();

            chaincodeStub.putState.onFirstCall().callsFake((key, value) => {
                if (!chaincodeStub.states) {
                    chaincodeStub.states = {};
                }
                chaincodeStub.states[key] = 'non-json-value';
            });

            await assetTransfer.CreateAsset(transactionContext, 'asset1', 'blue', 5, 'Robert', 100);
            await assetTransfer.CreateAsset(transactionContext, 'asset2', 'orange', 10, 'Paul', 200);
            await assetTransfer.CreateAsset(transactionContext, 'asset3', 'red', 15, 'Troy', 300);
            await assetTransfer.CreateAsset(transactionContext, 'asset4', 'pink', 20, 'Van', 400);

            let ret = await assetTransfer.GetAllAssets(transactionContext);
            ret = JSON.parse(ret);
            expect(ret.length).to.equal(4);

            let expected = [
                {Record: 'non-json-value'},
                {Record: {ID: 'asset2', Color: 'orange', Size: 10, Owner: 'Paul', AppraisedValue: 200}},
                {Record: {ID: 'asset3', Color: 'red', Size: 15, Owner: 'Troy', AppraisedValue: 300}},
                {Record: {ID: 'asset4', Color: 'pink', Size: 20, Owner: 'Van', AppraisedValue: 400}}
            ];

            expect(ret).to.eql(expected);
        });
    });
});
